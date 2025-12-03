# SPDX-License-Identifier: BSD-2-Clause
"""Vonage WebRTC transport."""

import asyncio
import itertools
import threading
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from loguru import logger

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    UserImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    import vonage_video_connector as vonage_video
    from vonage_video_connector.models import (
        AudioData,
        Connection,
        LoggingSettings,
        Publisher,
        PublisherAudioSettings,
        PublisherSettings,
        Session,
        SessionAudioSettings,
        SessionAVSettings,
        SessionSettings,
        SessionVideoPublisherSettings,
        Stream,
        Subscriber,
        SubscriberSettings,
        SubscriberVideoSettings,
        VideoFrame,
        VideoResolution,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        f"In order to use Vonage, you need to Vonage's native SDK wrapper for python installed."
    )
    raise Exception(f"Missing module: {e}")


class VonageVideoWebrtcTransportParams(TransportParams):
    """Parameters for the Vonage WebRTC transport.

    Parameters:
        publisher_name: Name of the publisher stream.
        publisher_enable_opus_dtx: Whether to enable OPUS DTX for publisher audio.
        session_enable_migration: Whether to enable session migration.
        audio_in_auto_subscribe: Whether to automatically subscribe to audio streams.
        video_in_auto_subscribe: Whether to automatically subscribe to video streams.
        video_in_preferred_width: Preferred width for video input capture.
        video_in_preferred_height: Preferred height for video input capture.
        video_in_preferred_framerate: Preferred framerate for video input capture.
    """

    publisher_name: str = ""
    publisher_enable_opus_dtx: bool = False
    session_enable_migration: bool = False
    audio_in_auto_subscribe: bool = True
    video_in_auto_subscribe: bool = False
    video_in_preferred_resolution: Optional[tuple[int, int]] = None
    video_in_preferred_framerate: Optional[int] = None


@dataclass
class SubscribeSettings:
    """Parameters for stream input subscription.

    Parameters:
        capture_audio: Whether to subscribe to audio.
        capture_video: Whether to subscribe to video.
        preferred_resolution: Preferred resolution for video subscription.
        preferred_framerate: Preferred framerate for video subscription.
    """

    subscribe_to_audio: bool = True
    subscribe_to_video: bool = False
    preferred_resolution: Optional[tuple[int, int]] = None
    preferred_framerate: Optional[int] = None


class VonageException(Exception):
    """Exception raised when a Vonage transport operation fails or encounters an error."""

    pass


async def async_noop(*args: Any, **kwargs: Any) -> None:
    """No operation async function."""
    pass


@dataclass
class VonageClientListener:
    """Listener for Vonage client events.

    Parameters:
        on_connected: Async callback when session is connected.
        on_disconnected: Async callback when session is disconnected.
        on_error: Async callback for session errors.
        on_audio_in: Async callback for incoming audio data.
        on_stream_received: Async callback when a stream is received.
        on_stream_dropped: Async callback when a stream is dropped.
        on_subscriber_connected: Async callback when a subscriber connects.
        on_subscriber_disconnected: Async callback when a subscriber disconnects.
    """

    on_connected: Callable[[Session], Awaitable[None]] = async_noop
    on_disconnected: Callable[[Session], Awaitable[None]] = async_noop
    on_error: Callable[[Session, str, int], Awaitable[None]] = async_noop
    on_audio_in: Callable[[Session, InputAudioRawFrame], Awaitable[None]] = async_noop
    on_stream_received: Callable[[Session, Stream], Awaitable[None]] = async_noop
    on_stream_dropped: Callable[[Session, Stream], Awaitable[None]] = async_noop
    on_subscriber_connected: Callable[[Subscriber], Awaitable[None]] = async_noop
    on_subscriber_disconnected: Callable[[Subscriber], Awaitable[None]] = async_noop
    on_subscriber_video_in: Callable[[Subscriber, VideoFrame], Awaitable[None]] = async_noop
    on_video_in: Callable[[Subscriber, UserImageRawFrame], Awaitable[None]] = async_noop


@dataclass
class AudioProps:
    """Audio properties for normalization.

    Parameters:
        sample_rate: The sample rate of the audio.
        is_stereo: Whether the audio is stereo (True) or mono (False).
    """

    sample_rate: int
    is_stereo: bool


VIDEO_CONNECTOR_TIMEOUT: timedelta = timedelta(seconds=30)
DEFAULT_SAMPLE_RATE: int = 48000

AUDIO_QUEUE_MAXSIZE: int = 500
VIDEO_QUEUE_MAXSIZE: int = 50

TA = TypeVar("TA", InputAudioRawFrame, OutputAudioRawFrame)
TV = TypeVar("TV", UserImageRawFrame, OutputImageRawFrame)
SimpleCoroutine = Coroutine[Any, Any, None]


PIPECAT_TO_VONAGE_FORMAT_MAP: dict[str, str] = {
    "YUV": "YUV420P",
    "RGB": "RGB24",
    "ARGB": "ARGB32",
}
VONAGE_TO_PIPECAT_FORMAT_MAP: dict[str, str] = {
    v: k for k, v in PIPECAT_TO_VONAGE_FORMAT_MAP.items()
}

DUMMY_CONNECTION = Connection(id="", creation_time=datetime.min)


class VonageClient:
    """Client for managing a Vonage Video session.

    Handles connection, publishing, subscribing, and event callbacks for a Vonage Video session.
    """

    def __init__(
        self,
        application_id: str,
        session_id: str,
        token: str,
        params: VonageVideoWebrtcTransportParams,
    ):
        """Initialize the Vonage client.

        Args:
            application_id: The Vonage Video application ID.
            session_id: The session ID to connect to.
            token: The authentication token for the session.
            params: Parameters to configure the Vonage client.
        """
        self._client = vonage_video.VonageVideoClient()
        self._application_id: str = application_id
        self._session_id: str = session_id
        self._token: str = token
        self._params = params.model_copy(
            # make sure we have auto-subscribe only if the respective media is enabled
            update={
                "audio_in_auto_subscribe": params.audio_in_auto_subscribe
                and params.audio_in_enabled,
                "video_in_auto_subscribe": params.video_in_auto_subscribe
                and params.video_in_enabled,
            }
        )
        # having these two settings separately to make them non-optional
        self._audio_in_sample_rate = params.audio_in_sample_rate or DEFAULT_SAMPLE_RATE
        self._audio_out_sample_rate = params.audio_out_sample_rate or DEFAULT_SAMPLE_RATE

        self._connected: bool = False
        self._connection_counter: int = 0
        self._connecting_future: Optional[asyncio.Future[None]] = None
        self._disconnecting_future: Optional[asyncio.Future[None]] = None

        self._listener_id_gen: itertools.count[int] = itertools.count()
        self._listeners: dict[int, VonageClientListener] = {}

        self._publisher: Optional[Publisher] = None
        self._session = Session(id=session_id)

        self._resampler = create_stream_resampler()

        self._task_manager: Optional[BaseTaskManager] = None
        self._loop_thread_id = threading.get_ident()
        self._event_queue: Optional[asyncio.Queue[SimpleCoroutine]] = None
        self._event_task: Optional[asyncio.Task[None]] = None
        self._audio_queue: Optional[asyncio.Queue[SimpleCoroutine]] = None
        self._audio_task: Optional[asyncio.Task[None]] = None
        self._video_queue: Optional[asyncio.Queue[SimpleCoroutine]] = None
        self._video_task: Optional[asyncio.Task[None]] = None

        # used for blocking calls to connect and disconnect
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._session_streams: dict[str, Stream] = {}
        self._session_subscriptions: dict[str, SubscribeSettings] = {}

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Setup the client with task manager and event queues.

        Args:
            setup: The frame processor setup configuration.
        """
        if self._task_manager:
            return

        self._task_manager = setup.task_manager

        # tasks from the generic event queue should allow concurrent processing as they
        # may await on new events posted to the same queue
        self._event_queue = asyncio.Queue()
        self._event_task = self._task_manager.create_task(
            self._sdk_cb_to_loop_task_handler(self._event_queue, allow_concurrent=True),
            f"event_callback_task",
        )
        # audio and video tasks should be processed one at a time
        self._audio_queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)
        self._audio_task = self._task_manager.create_task(
            self._sdk_cb_to_loop_task_handler(self._audio_queue, allow_concurrent=False),
            f"audio_callback_task",
        )
        self._video_queue = asyncio.Queue(maxsize=VIDEO_QUEUE_MAXSIZE)
        self._video_task = self._task_manager.create_task(
            self._sdk_cb_to_loop_task_handler(self._video_queue, allow_concurrent=False),
            f"video_callback_task",
        )

    async def cleanup(self) -> None:
        """Cleanup the client, disconnecting if necessary."""
        if self._connected:
            await self.disconnect()

        if self._event_task and self._task_manager:
            await self._task_manager.cancel_task(self._event_task)
            await self._event_task
            self._event_task = None
        if self._audio_task and self._task_manager:
            await self._task_manager.cancel_task(self._audio_task)
            await self._audio_task
            self._audio_task = None
        if self._video_task and self._task_manager:
            await self._task_manager.cancel_task(self._video_task)
            await self._video_task
            self._video_task = None

    def add_listener(self, listener: VonageClientListener) -> int:
        """Add a listener to the Vonage client.

        Args:
            listener: The VonageClientListener to add.

        Returns:
            The unique ID assigned to the listener.
        """
        listener_id = next(self._listener_id_gen)
        self._listeners[listener_id] = listener
        return listener_id

    def remove_listener(self, listener_id: int) -> None:
        """Remove a listener from the Vonage client.

        Args:
            listener_id: The ID of the listener to remove.
        """
        self._listeners.pop(listener_id, None)

    async def connect(self) -> None:
        """Connect to the Vonage session."""
        logger.info(f"Connecting with session string {self._session_id}")

        if self._disconnecting_future is not None:
            logger.info(
                f"Waiting for disconnection to complete before connecting to {self._session_id}"
            )
            await self._disconnecting_future

        if self._connected:
            logger.info(f"Already connected to {self._session_id}")
            self._connection_counter += 1
            return

        if self._connecting_future is not None:
            logger.info(f"Already connecting to {self._session_id}")

            # if we already connecting, await for the publish ready event
            await self._connecting_future
            self._connection_counter += 1
            return

        # this future will allow concurrent calls to connect to wait until the first connect call is done
        self._connecting_future = self._get_event_loop().create_future()

        try:
            await self._sdk_connect()
        except Exception:
            future = self._connecting_future
            self._connecting_future = None
            future.cancel()
            raise

        logger.info(f"Connected to {self._session_id}")
        self._connected = True
        self._connection_counter += 1

        # all concurrent calls to connect can now proceed
        future = self._connecting_future
        self._connecting_future = None
        future.set_result(None)

        await asyncio.gather(
            *(listener.on_connected(self._session) for listener in self._listeners.values())
        )

    async def disconnect(self) -> None:
        """Disconnect from the Vonage session."""
        if self._connecting_future is not None:
            logger.info(
                f"Waiting for connection to complete before disconnecting from {self._session_id}"
            )
            await self._connecting_future

        if not self._connected:
            logger.info(f"Already disconnected from {self._session_id}")
            return

        self._connection_counter -= 1
        if self._connection_counter != 0:
            logger.info(
                f"{self._connection_counter} connections still active for {self._session_id}"
            )
            return

        self._disconnecting_future = self._get_event_loop().create_future()

        logger.info(f"Disconnecting from {self._session_id}")

        # ensure we clear up any pending SDK callback events and media buffers
        if self._event_queue:
            self._clear_queue(self._event_queue)
        self.clear_media_buffers()
        self._session_streams.clear()
        self._session_subscriptions.clear()

        try:
            await self._sdk_disconnect()
        except Exception:
            future = self._disconnecting_future
            self._disconnecting_future = None
            self._connection_counter += 1
            future.cancel()
            raise

        logger.info(f"Disconnected from {self._session_id}")

        self._connected = False

        future = self._disconnecting_future
        self._disconnecting_future = None
        future.set_result(None)

        await asyncio.gather(
            *(listener.on_disconnected(self._session) for listener in self._listeners.values())
        )

    def clear_media_buffers(self) -> None:
        """Clear output media buffers in the Vonage session."""
        logger.debug(f"Clearing media buffers {self._session_id}")
        if self._audio_queue:
            self._clear_queue(self._audio_queue)
        if self._video_queue:
            self._clear_queue(self._video_queue)
        self._client.clear_media_buffers()

    async def write_audio(self, audio_frame: OutputAudioRawFrame) -> bool:
        """Write audio data to the Vonage session.

        Args:
            audio_frame: Audio frame to write
        """
        target_audio_props = AudioProps(
            sample_rate=self._audio_out_sample_rate,
            is_stereo=self._params.audio_out_channels == 2,
        )
        proc_audio_frame = await self._process_audio_if_needed(audio_frame, target_audio_props)

        return self._client.add_audio(
            AudioData(
                sample_buffer=memoryview(proc_audio_frame.audio).cast("h"),
                number_of_frames=proc_audio_frame.num_frames,
                number_of_channels=self._params.audio_out_channels,
                sample_rate=self._audio_out_sample_rate,
            )
        )

    async def subscribe_to_stream(self, stream_id: str, params: SubscribeSettings) -> None:
        """Subscribe to a participant's stream.

        Args:
            stream_id: The ID of the participant to subscribe to.
            params: Subscription parameters for the subscription.
        """
        stream = self._session_streams.get(stream_id, None) or Stream(
            id=stream_id, connection=DUMMY_CONNECTION
        )

        await self._sdk_subscribe(stream, params)

    @staticmethod
    def vonage_image_format(pipecat_format: str) -> str:
        """Normalize Pipecat image format to Vonage format."""
        return PIPECAT_TO_VONAGE_FORMAT_MAP.get(pipecat_format, pipecat_format)

    @staticmethod
    def pipecat_image_format(vonage_format: str) -> str:
        """Normalize Vonage image format to Pipecat format."""
        return VONAGE_TO_PIPECAT_FORMAT_MAP.get(vonage_format, vonage_format)

    async def write_video(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the transport.

        Args:
            frame: The output video frame to write.
        """
        if not self._check_image_data(frame):
            return False

        processed_frame = self._process_video_if_needed(frame)

        return self._client.add_video(
            VideoFrame(
                frame_buffer=memoryview(processed_frame.image).cast("B"),
                resolution=VideoResolution(
                    width=processed_frame.size[0], height=processed_frame.size[1]
                ),
                format=self.vonage_image_format(
                    processed_frame.format or self._params.video_out_color_format
                ),
            )
        )

    def _check_image_data(self, frame: OutputImageRawFrame) -> bool:
        """Check the image data for validity.

        Args:
            frame: The OutputImageRawFrame to check.
        """
        res = True
        if frame.format != self._params.video_out_color_format:
            logger.error(
                f"Expected color format {self._params.video_out_color_format}, got {frame.format}"
            )
            res = False
        if (
            frame.size[0] != self._params.video_out_width
            or frame.size[1] != self._params.video_out_height
        ):
            logger.error(
                f"Expected resolution {self._params.video_out_width}x{self._params.video_out_height}, "
                f"got {frame.size[0]}x{frame.size[1]}"
            )
            res = False
        return res

    async def _sdk_connect(self) -> None:
        # this future will be set when the session is ready to publish, audio needs a special callback before
        # publishing
        ready_to_publish_future = self._get_event_loop().create_future()

        # this callback will be called when the session is ready to publish audio, video-only doesn't need it
        def audio_ready_cb(session: Session) -> None:
            async def async_cb() -> None:
                logger.info(f"Session {session.id} ready to publish")
                ready_to_publish_future.set_result(None)

            self._sdk_event_cb_to_loop(async_cb())

        def connect_proc() -> None:
            if not self._client.connect(
                application_id=self._application_id,
                session_id=self._session_id,
                token=self._token,
                session_settings=SessionSettings(
                    av=SessionAVSettings(
                        audio_publisher=SessionAudioSettings(
                            sample_rate=self._audio_out_sample_rate,
                            number_of_channels=self._params.audio_out_channels,
                        ),
                        audio_subscribers_mix=SessionAudioSettings(
                            sample_rate=self._audio_in_sample_rate,
                            number_of_channels=self._params.audio_in_channels,
                        ),
                        video_publisher=SessionVideoPublisherSettings(
                            resolution=VideoResolution(
                                width=self._params.video_out_width,
                                height=self._params.video_out_height,
                            ),
                            fps=self._params.video_out_framerate,
                            format=self.vonage_image_format(self._params.video_out_color_format),
                        ),
                    ),
                    enable_migration=self._params.session_enable_migration,
                    logging=LoggingSettings(level="INFO"),
                ),
                on_error_cb=self._on_session_error_cb,
                on_connected_cb=self._on_session_connected_cb,
                on_disconnected_cb=self._on_session_disconnected_cb,
                on_stream_received_cb=self._on_stream_received_cb,
                on_stream_dropped_cb=self._on_stream_dropped_cb,
                on_audio_data_cb=self._on_session_audio_data_cb,
                on_ready_for_audio_cb=audio_ready_cb,
            ):
                logger.error(f"Could not connect to {self._session_id}")
                raise VonageException("Could not connect to session")

        async def async_proc() -> None:
            await self._get_event_loop().run_in_executor(self._executor, connect_proc)

            # when audio publishing is enabled we need to wait for the session to be ready for audio
            # however, if only video is being published at this point we don't need to wait anymore
            if self._params.audio_out_enabled:
                logger.info(f"Waiting for {self._session_id} to be ready to publish audio")
                await ready_to_publish_future

        try:
            await asyncio.wait_for(async_proc(), timeout=VIDEO_CONNECTOR_TIMEOUT.total_seconds())
        except asyncio.TimeoutError as exc:
            logger.error(f"Timeout connecting to Vonage session {self._session_id}")

            raise exc

    async def _sdk_disconnect(self) -> None:
        def disconnect_proc() -> None:
            if self._publisher:
                self._client.unpublish()
                self._publisher = None
            self._client.disconnect()

        try:
            await asyncio.wait_for(
                self._get_event_loop().run_in_executor(self._executor, disconnect_proc),
                timeout=VIDEO_CONNECTOR_TIMEOUT.total_seconds(),
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout disconnecting from Vonage session {self._session_id}")
            raise

    async def _sdk_subscribe(self, stream: Stream, params: SubscribeSettings) -> None:
        subscribed_future = self._get_event_loop().create_future()
        self._session_subscriptions[stream.id] = params

        def on_error_cb(subscriber: Subscriber, error: str, code: int) -> None:
            async def async_cb() -> None:
                logger.error(f"Subscriber {subscriber.stream.id} error: {error} (code {code})")
                self._session_subscriptions.pop(subscriber.stream.id, None)
                if not subscribed_future.done():
                    subscribed_future.set_exception(
                        VonageException(f"Subscriber error: {error} (code {code})")
                    )

            self._sdk_event_cb_to_loop(async_cb())

        def on_connected_cb(subscriber: Subscriber) -> None:
            async def async_cb() -> None:
                logger.info(f"Subscriber {subscriber.stream.id} connected")

                if not subscribed_future.done():
                    subscribed_future.set_result(None)

                await asyncio.gather(
                    *(
                        listener.on_subscriber_connected(subscriber)
                        for listener in self._listeners.values()
                    )
                )

            self._sdk_event_cb_to_loop(async_cb())

        def on_subscriber_disconnected_cb(subscriber: Subscriber) -> None:
            async def async_cb() -> None:
                logger.info(
                    f"Subscriber disconnected session={self._session_id} subscriber={subscriber.stream.id} "
                )
                self._session_subscriptions.pop(subscriber.stream.id, None)
                if not subscribed_future.done():
                    subscribed_future.set_exception(
                        VonageException(
                            f"Subscriber {subscriber.stream.id} disconnected before connecting"
                        )
                    )
                await asyncio.gather(
                    *(
                        listener.on_subscriber_disconnected(subscriber)
                        for listener in self._listeners.values()
                    )
                )

            self._sdk_event_cb_to_loop(async_cb())

        async def process() -> None:
            logger.info(
                f"Subscribing to stream {stream.id} audio={params.subscribe_to_audio} "
                f"video={params.subscribe_to_video}"
            )
            if not self._client.subscribe(
                stream,
                settings=SubscriberSettings(
                    subscribe_to_audio=params.subscribe_to_audio,
                    subscribe_to_video=params.subscribe_to_video,
                    video_settings=SubscriberVideoSettings(
                        preferred_resolution=(
                            VideoResolution(
                                width=params.preferred_resolution[0],
                                height=params.preferred_resolution[1],
                            )
                            if params.preferred_resolution
                            else None
                        ),
                        preferred_framerate=params.preferred_framerate,
                    ),
                ),
                on_error_cb=on_error_cb,
                on_connected_cb=on_connected_cb,
                on_disconnected_cb=on_subscriber_disconnected_cb,
                on_render_frame_cb=self._on_subscriber_video_data_cb,
            ):
                subscribed_future.cancel()
                raise VonageException(f"Could not subscribe to stream {stream.id}")

            await subscribed_future

        try:
            await asyncio.wait_for(process(), timeout=VIDEO_CONNECTOR_TIMEOUT.total_seconds())
        except asyncio.TimeoutError:
            logger.error(f"Timeout subscribing to Vonage stream {stream.id}")
            self._session_subscriptions.pop(stream.id, None)
            raise

    @staticmethod
    def _clear_queue(queue: asyncio.Queue[SimpleCoroutine]) -> None:
        """Clear all items from the given asyncio queue."""
        try:
            while True:
                item = queue.get_nowait()
                # Close coroutines to avoid "never awaited" warnings
                item.close()
                queue.task_done()
        except asyncio.QueueEmpty:
            pass

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop from the task manager."""
        if not self._task_manager:
            raise Exception(f"{self}: missing task manager (pipeline not started?)")
        return self._task_manager.get_event_loop()

    async def _sdk_cb_to_loop_task_handler(
        self, queue: asyncio.Queue[SimpleCoroutine], allow_concurrent: bool
    ) -> None:
        """Read coroutines generated from SDK callbacks in the given queue executing them in the event loop."""
        # ensure we know the thread id of the event loop
        self._loop_thread_id = threading.get_ident()
        # if we allow concurrent tasks, process them as they come in
        if allow_concurrent:
            active_tasks = set()

            async def wrapped_task(coroutine: SimpleCoroutine) -> None:
                try:
                    await coroutine
                except Exception as exc:
                    logger.error(f"Exception in SDK callback task: {exc}")
                finally:
                    active_tasks.discard(task)
                    queue.task_done()

            try:
                while True:
                    async_task = await queue.get()
                    task = asyncio.create_task(wrapped_task(async_task))
                    active_tasks.add(task)
            except asyncio.CancelledError:
                # Cancel all active tasks
                for task in active_tasks:
                    task.cancel()

                # Wait for them to finish cancelling
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
        # if we only allow one task at a time, process them sequentially
        else:
            while True:
                try:
                    async_task = await queue.get()
                    await async_task
                    queue.task_done()
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(f"Exception in SDK callback task: {exc}")

    def _sdk_event_cb_to_loop(self, callback: SimpleCoroutine) -> None:
        """From an SDK thread queue an event coroutine to be asynchronously executed in the task manager event loop."""
        self._sdk_cb_to_loop("event", self._event_queue, callback)

    def _sdk_audio_cb_to_loop(self, callback: SimpleCoroutine) -> None:
        """From an SDK thread queue an audio coroutine to be asynchronously executed in the task manager event loop."""
        self._sdk_cb_to_loop("audio", self._audio_queue, callback)

    def _sdk_video_cb_to_loop(self, callback: SimpleCoroutine) -> None:
        """From an SDK thread queue a video coroutine to be asynchronously executed in the task manager event loop."""
        self._sdk_cb_to_loop("video", self._video_queue, callback)

    def _sdk_cb_to_loop(
        self,
        queue_type_name: str,
        queue: Optional[asyncio.Queue[SimpleCoroutine]],
        async_task: SimpleCoroutine,
    ) -> None:
        """From an SDK thread queue a coroutine to be asynchronously executed in the task manager event loop.

        If the coroutine queue is full the event will be dropped and a warning logged.
        """
        if not queue:
            raise Exception(f"missing {queue_type_name} queue (pipeline not started?)")

        def put_coroutine() -> None:
            try:
                queue.put_nowait(async_task)
            except asyncio.QueueFull:
                logger.warning(
                    f"{queue_type_name} queue is full, dropping SDK {queue_type_name} callback."
                )
                async_task.close()

        if threading.get_ident() == self._loop_thread_id:
            put_coroutine()
        else:
            self._get_event_loop().call_soon_threadsafe(put_coroutine)

    def _on_session_error_cb(self, session: Session, description: str, code: int) -> None:
        async def async_cb() -> None:
            logger.warning(f"Session error {session.id} code={code} description={description}")
            await asyncio.gather(
                *(
                    listener.on_error(session, description, code)
                    for listener in self._listeners.values()
                )
            )

        self._sdk_event_cb_to_loop(async_cb())

    def _on_session_connected_cb(self, session: Session) -> None:
        async def async_cb() -> None:
            logger.info(f"Session connected {session.id}")
            self._session = session
            if self._params.audio_out_enabled or self._params.video_out_enabled:
                logger.info(
                    f"Publishing audio={self._params.audio_out_enabled} video={self._params.video_out_enabled} "
                    f"for session {session.id}"
                )
                # TODO this could be run in the executor pool as it blocks
                self._client.publish(
                    settings=PublisherSettings(
                        name=self._params.publisher_name,
                        audio_settings=PublisherAudioSettings(
                            enable_stereo_mode=self._params.audio_out_channels == 2,
                            enable_opus_dtx=self._params.publisher_enable_opus_dtx,
                        ),
                        has_audio=self._params.audio_out_enabled,
                        has_video=self._params.video_out_enabled,
                    ),
                    on_error_cb=self._on_publisher_error_cb,
                    on_stream_created_cb=self._on_publisher_stream_created_cb,
                    on_stream_destroyed_cb=self._on_publisher_stream_destroyed_cb,
                )
            else:
                logger.info(f"No audio or video to publish for session {session.id}")

        self._sdk_event_cb_to_loop(async_cb())

    def _on_session_disconnected_cb(self, session: Session) -> None:
        async def async_cb() -> None:
            logger.info(f"Session disconnected {session.id}")
            self._connected = False

        self._sdk_event_cb_to_loop(async_cb())

    def _on_publisher_error_cb(self, publisher: Publisher, description: str, code: int) -> None:
        async def async_cb() -> None:
            logger.warning(
                f"Publisher error session={self._session_id} publisher={publisher.stream.id} "
                f"code={code} description={description}"
            )

        self._sdk_event_cb_to_loop(async_cb())

    def _on_publisher_stream_created_cb(self, publisher: Publisher) -> None:
        async def async_cb() -> None:
            logger.info(
                f"Publisher stream created session={self._session_id} publisher={publisher.stream.id}"
            )
            self._publisher = publisher

        self._sdk_event_cb_to_loop(async_cb())

    def _on_publisher_stream_destroyed_cb(self, publisher: Publisher) -> None:
        async def async_cb() -> None:
            logger.info(
                f"Publisher stream destroyed session={self._session_id} publisher={publisher.stream.id}"
            )

        self._sdk_event_cb_to_loop(async_cb())

    def _on_session_audio_data_cb(self, session: Session, audio_data: AudioData) -> None:
        """Callback for incoming mixed audio data for all the subscribers in the session."""
        # we need to keep a copy of the audio data as it is a memory view and it will be lost when processed async later
        audio_frame = InputAudioRawFrame(
            audio=audio_data.sample_buffer.tobytes(),
            sample_rate=audio_data.sample_rate,
            num_channels=audio_data.number_of_channels,
        )

        async def async_cb() -> None:
            target_audio_props = AudioProps(
                sample_rate=self._audio_in_sample_rate,
                is_stereo=self._params.audio_in_channels == 2,
            )
            proc_audio_frame = await self._process_audio_if_needed(audio_frame, target_audio_props)

            await asyncio.gather(
                *(
                    listener.on_audio_in(session, proc_audio_frame)
                    for listener in self._listeners.values()
                )
            )

        self._sdk_audio_cb_to_loop(async_cb())

    def _on_stream_received_cb(self, session: Session, stream: Stream) -> None:
        async def async_cb() -> None:
            logger.info(f"Stream received session={session.id} stream={stream.id}")
            self._session_streams[stream.id] = stream

            # raise the event before auto subscribing so listeners can decide what to do
            await asyncio.gather(
                *(
                    listener.on_stream_received(session, stream)
                    for listener in self._listeners.values()
                )
            )

            # if we have auto-subscribe enabled, subscribe to the stream if it hasn't been subscribed yet
            auto_subscribe = (
                self._params.audio_in_auto_subscribe or self._params.video_in_auto_subscribe
            )
            if auto_subscribe and not stream.id in self._session_subscriptions:
                await self._sdk_subscribe(
                    stream,
                    SubscribeSettings(
                        subscribe_to_audio=self._params.audio_in_auto_subscribe,
                        subscribe_to_video=self._params.video_in_auto_subscribe,
                        preferred_resolution=(self._params.video_in_preferred_resolution),
                        preferred_framerate=self._params.video_in_preferred_framerate,
                    ),
                )

        self._sdk_event_cb_to_loop(async_cb())

    def _on_stream_dropped_cb(self, session: Session, stream: Stream) -> None:
        async def async_cb() -> None:
            logger.info(f"Stream dropped session={session.id} stream={stream.id}")
            if stream.id in self._session_subscriptions:
                self._client.unsubscribe(stream)
                self._session_subscriptions.pop(stream.id, None)
            self._session_streams.pop(stream.id, None)

            await asyncio.gather(
                *(
                    listener.on_stream_dropped(session, stream)
                    for listener in self._listeners.values()
                )
            )

        self._sdk_event_cb_to_loop(async_cb())

    def _on_subscriber_video_data_cb(self, subscriber: Subscriber, frame: VideoFrame) -> None:
        """Callback for incoming per stream data for all the subscribers in the session."""
        # we need to keep a copy of the audio data as it is a memory view and it will be lost when processed async later
        pipecat_frame = UserImageRawFrame(
            user_id=subscriber.stream.id,
            image=frame.frame_buffer.tobytes(),
            size=(frame.resolution.width, frame.resolution.height),
            format=self.pipecat_image_format(frame.format),
        )

        async def async_cb() -> None:
            processed_frame = self._process_video_if_needed(pipecat_frame)

            await asyncio.gather(
                *(
                    listener.on_video_in(subscriber, processed_frame)
                    for listener in self._listeners.values()
                )
            )

        self._sdk_video_cb_to_loop(async_cb())

    async def _process_audio_if_needed(self, audio_frame: TA, target_props: AudioProps) -> TA:
        check_audio_data(audio_frame.audio, audio_frame.num_frames, audio_frame.num_channels)

        current_audio_props = AudioProps(
            sample_rate=audio_frame.sample_rate,
            is_stereo=audio_frame.num_channels == 2,
        )
        if current_audio_props != target_props:
            audio_np = np.frombuffer(audio_frame.audio, dtype=np.int16)
            processed_audio_np = await process_audio(
                self._resampler,
                audio_np,
                current_audio_props,
                target_props,
            )

            processed_audio_frame = replace(
                audio_frame,
                audio=processed_audio_np.tobytes(),
                sample_rate=target_props.sample_rate,
                num_channels=2 if target_props.is_stereo else 1,
            )
            return processed_audio_frame
        else:
            return audio_frame

    @staticmethod
    def _process_video_if_needed(frame: TV) -> TV:
        # this will switch RGB to BGR or BGR to RGB as needed
        # video coming from Vonage comes as BGR when using RGB format, whereas Pipecat uses RGB
        video_bytes = frame.image
        if frame.format == "RGB":
            np_rgb = np.frombuffer(frame.image, dtype=np.uint8)
            np_bgr = np_rgb.reshape(frame.size[1], frame.size[0], 3)[:, :, ::-1]
            bgr_bytes = np_bgr.tobytes()
            video_bytes = bgr_bytes
        # TODO(Toni S.): do we need to do any conversion for ARGB?

        return replace(frame, image=video_bytes)


class VonageVideoWebrtcInputTransport(BaseInputTransport):
    """Input transport for Vonage, handling audio input from the Vonage session.

    Receives audio from a Vonage Video session and pushes it as input frames.
    """

    _params: VonageVideoWebrtcTransportParams

    def __init__(self, client: VonageClient, params: VonageVideoWebrtcTransportParams):
        """Initialize the Vonage input transport.

        Args:
            client: The VonageClient instance to use.
            params: Transport parameters for input configuration.
        """
        super().__init__(params)
        self._initialized: bool = False
        self._client: VonageClient = client
        self._listener_id: int = -1
        self._connected: bool = False

    async def start(self, frame: StartFrame) -> None:
        """Start the Vonage input transport.

        Args:
            frame: The StartFrame to initiate the transport.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.audio_in_enabled or self._params.video_in_enabled:
            self._listener_id = self._client.add_listener(
                VonageClientListener(on_audio_in=self._audio_in_cb, on_video_in=self._video_in_cb)
            )
            await self._client.connect()
            self._connected = True

        await self.set_transport_ready(frame)

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self) -> None:
        """Cleanup input transport."""
        await super().cleanup()  # type: ignore
        await self._client.cleanup()

    async def _audio_in_cb(self, _session: Session, audio: InputAudioRawFrame) -> None:
        if self._connected and self._params.audio_in_enabled:
            await self.push_audio_frame(audio)

    async def _video_in_cb(self, _subscriber: Subscriber, video: UserImageRawFrame) -> None:
        if self._connected and self._params.video_in_enabled:
            await self.push_video_frame(video)

    async def stop(self, frame: EndFrame) -> None:
        """Stop the Vonage input transport.

        Args:
            frame: The EndFrame to stop the transport.
        """
        await super().stop(frame)
        if self._connected:
            self._client.remove_listener(self._listener_id)
            self._connected = False
            await self._client.disconnect()

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel the Vonage input transport.

        Args:
            frame: The CancelFrame to cancel the transport.
        """
        await super().cancel(frame)
        if self._connected:
            self._client.remove_listener(self._listener_id)
            self._connected = False
            await self._client.disconnect()

    async def subscribe_to_stream(self, stream_id: str, params: SubscribeSettings) -> None:
        """Subscribe to a participant's stream.

        Args:
            stream_id: The ID of the participant to subscribe to.
            params: Subscription parameters for the subscription.
        """
        await self._client.subscribe_to_stream(stream_id, params)


class VonageVideoWebrtcOutputTransport(BaseOutputTransport):
    """Output transport for Vonage, handling audio output to the Vonage session.

    Sends audio frames to a Vonage Video session as output.
    """

    _params: VonageVideoWebrtcTransportParams

    def __init__(self, client: VonageClient, params: VonageVideoWebrtcTransportParams):
        """Initialize the Vonage output transport.

        Args:
            client: The VonageClient instance to use.
            params: Transport parameters for output configuration.
        """
        super().__init__(params)
        self._initialized: bool = False
        self._client = client
        self._connected: bool = False

    async def start(self, frame: StartFrame) -> None:
        """Start the Vonage output transport.

        Args:
            frame: The StartFrame to initiate the transport.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.audio_out_enabled or self._params.video_out_enabled:
            await self._client.connect()
            self._connected = True

        await self.set_transport_ready(frame)

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self) -> None:
        """Cleanup output transport."""
        await super().cleanup()  # type: ignore
        await self._client.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process a frame for the Vonage output transport.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # if we get an interruption frame, we need to ensure the buffers inside Vonage Video Connector are cleared
        if self._connected and isinstance(frame, InterruptionFrame):
            logger.info("Clearing Vonage media buffers due to interruption frame")
            self._client.clear_media_buffers()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the Vonage session.

        Args:
            frame: The OutputAudioRawFrame to send.
        """
        result = False
        if self._connected and self._params.audio_out_enabled:
            result = await self._client.write_audio(frame)

        return result

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the transport.

        Args:
            frame: The output video frame to write.
        """
        result = False
        if self._connected and self._params.video_out_enabled:
            result = await self._client.write_video(frame)

        return result

    async def stop(self, frame: EndFrame) -> None:
        """Stop the Vonage output transport.

        Args:
            frame: The EndFrame to stop the transport.
        """
        await super().stop(frame)
        if self._connected:
            self._connected = False
            await self._client.disconnect()

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel the Vonage output transport.

        Args:
            frame: The CancelFrame to cancel the transport.
        """
        await super().cancel(frame)
        if self._connected:
            self._connected = False
            await self._client.disconnect()


class VonageVideoWebrtcTransport(BaseTransport):
    """Vonage WebRTC transport implementation for Pipecat.

    Provides input and output audio transport for Vonage Video sessions, supporting event handling
    for session and participant lifecycle.

    Supported features:

    - Audio input and output transport for Vonage Video sessions
    - Event handler registration for session and participant events
    - Publisher and subscriber management
    - Configurable audio and migration parameters
    """

    _params: VonageVideoWebrtcTransportParams

    def __init__(
        self,
        application_id: str,
        session_id: str,
        token: str,
        params: VonageVideoWebrtcTransportParams,
    ):
        """Initialize the Vonage WebRTC transport.

        Args:
            application_id: The Vonage Video application ID.
            session_id: The session ID to connect to.
            token: The authentication token for the session.
            params: Transport parameters for input/output configuration.
        """
        super().__init__()
        self._params = params

        self._client = VonageClient(application_id, session_id, token, params)

        # Register supported handlers.
        self._register_event_handler("on_joined")
        self._register_event_handler("on_left")
        self._register_event_handler("on_error")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")

        self._client.add_listener(
            VonageClientListener(
                on_connected=self._on_connected,
                on_disconnected=self._on_disconnected,
                on_error=self._on_error,
                on_stream_received=self._on_stream_received,
                on_stream_dropped=self._on_stream_dropped,
                on_subscriber_connected=self._on_subscriber_connected,
                on_subscriber_disconnected=self._on_subscriber_disconnected,
            )
        )

        self._input: Optional[VonageVideoWebrtcInputTransport] = None
        self._output: Optional[VonageVideoWebrtcOutputTransport] = None
        self._one_stream_received: bool = False

    def input(self) -> FrameProcessor:
        """Get the input transport for Vonage.

        Returns:
            The VonageVideoWebrtcInputTransport instance.
        """
        if not self._input:
            self._input = VonageVideoWebrtcInputTransport(self._client, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        """Get the output transport for Vonage.

        Returns:
            The VonageVideoWebrtcOutputTransport instance.
        """
        if not self._output:
            self._output = VonageVideoWebrtcOutputTransport(self._client, self._params)
        return self._output

    async def subscribe_to_stream(self, stream_id: str, params: SubscribeSettings) -> None:
        """Subscribe to a participant's stream.

        Args:
            stream_id: The ID of the participant to subscribe to.
            params: Subscription parameters for the subscription.
        """
        if self._input:
            await self._input.subscribe_to_stream(stream_id, params)

    async def _on_connected(self, session: Session) -> None:
        """Handle session connected event.

        Args:
            session: The connected Session object.
        """
        await self._call_event_handler("on_joined", {"sessionId": session.id})

    async def _on_disconnected(self, _session_id: Session) -> None:
        """Handle session disconnected event.

        Args:
            _session_id: The disconnected Session object.
        """
        await self._call_event_handler("on_left")

    async def _on_error(self, _session: Session, description: str, _code: int) -> None:
        """Handle session error event.

        Args:
            _session: The Session object.
            description: Error description.
            _code: Error code.
        """
        await self._call_event_handler("on_error", description)

    async def _on_stream_received(self, session: Session, stream: Stream) -> None:
        """Handle stream received event.

        Args:
            session: The Session object.
            stream: The received Stream object.
        """
        if not self._one_stream_received:
            self._one_stream_received = True
            await self._call_event_handler(
                "on_first_participant_joined", {"sessionId": session.id, "streamId": stream.id}
            )

        await self._call_event_handler(
            "on_participant_joined", {"sessionId": session.id, "streamId": stream.id}
        )

    async def _on_stream_dropped(self, session: Session, stream: Stream) -> None:
        """Handle stream dropped event.

        Args:
            session: The Session object.
            stream: The dropped Stream object.
        """
        await self._call_event_handler(
            "on_participant_left", {"sessionId": session.id, "streamId": stream.id}
        )

    async def _on_subscriber_connected(self, subscriber: Subscriber) -> None:
        """Handle subscriber connected event.

        Args:
            subscriber: The connected Subscriber object.
        """
        await self._call_event_handler(
            "on_client_connected", {"subscriberId": subscriber.stream.id}
        )

    async def _on_subscriber_disconnected(self, subscriber: Subscriber) -> None:
        """Handle subscriber disconnected event.

        Args:
            subscriber: The disconnected Subscriber object.
        """
        await self._call_event_handler(
            "on_client_disconnected", {"subscriberId": subscriber.stream.id}
        )


def check_audio_data(
    buffer: bytes | memoryview, number_of_frames: int, number_of_channels: int
) -> None:
    """Check the audio sample width based on buffer size, number of frames and channels."""
    if number_of_channels not in (1, 2):
        raise ValueError(f"We only accept mono or stereo audio, got {number_of_channels}")

    if isinstance(buffer, memoryview):
        bytes_per_sample = buffer.itemsize
    else:
        bytes_per_sample = len(buffer) // (number_of_frames * number_of_channels)

    if bytes_per_sample != 2:
        raise ValueError(f"We only accept 16 bit PCM audio, got {bytes_per_sample * 8} bit")


def process_audio_channels(
    audio: npt.NDArray[np.int16], current: AudioProps, target: AudioProps
) -> npt.NDArray[np.int16]:
    """Normalize audio channels to the target properties."""
    if current.is_stereo != target.is_stereo:
        if target.is_stereo:
            audio = np.repeat(audio, 2)
        else:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

    return audio


async def process_audio(
    resampler: BaseAudioResampler,
    audio: npt.NDArray[np.int16],
    current: AudioProps,
    target: AudioProps,
) -> npt.NDArray[np.int16]:
    """Normalize audio to the target properties."""
    res_audio = audio
    if current.sample_rate != target.sample_rate:
        # first normalize channels to mono if needed, then resample, then normalize channels to target
        res_audio = process_audio_channels(res_audio, current, replace(current, is_stereo=False))
        current = replace(current, is_stereo=False)
        res_audio_bytes: bytes = await resampler.resample(
            res_audio.tobytes(), current.sample_rate, target.sample_rate
        )
        res_audio = np.frombuffer(res_audio_bytes, dtype=np.int16)

    res_audio = process_audio_channels(res_audio, current, target)

    return res_audio
