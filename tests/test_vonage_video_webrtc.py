# SPDX-License-Identifier: BSD 2-Clause License

import asyncio
import inspect
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Optional
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, call, patch

import numpy as np
import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

# Mock the vonage_video module since it's not available in test environment
vonage_video_mock = MagicMock()
vonage_video_mock.VonageVideoClient = MagicMock()
vonage_video_mock.models = MagicMock()


# Create mock classes that match the expected interface
class MockAudioData:
    def __init__(
        self,
        sample_buffer: memoryview,
        number_of_frames: int,
        number_of_channels: int,
        sample_rate: int,
    ):
        self.sample_buffer = sample_buffer
        self.number_of_frames = number_of_frames
        self.number_of_channels = number_of_channels
        self.sample_rate = sample_rate


class MockSession:
    def __init__(self, id: str = "test_session"):
        self.id = id


class MockStream:
    def __init__(self, id: str = "test_stream"):
        self.id = id


class MockPublisher:
    def __init__(self, stream: Optional[MockStream] = None):
        self.stream = stream or MockStream()


class MockSubscriber:
    def __init__(self, stream: Optional[MockStream] = None):
        self.stream = stream or MockStream()


@dataclass(eq=True, frozen=True)
class MockSessionAudioSettings:
    sample_rate: int = 48000
    number_of_channels: int = 2


@dataclass(eq=True, frozen=True)
class MockVideoResolution:
    width: int = 1280
    height: int = 720


@dataclass(eq=True, frozen=True)
class MockSessionVideoPublisherSettings:
    resolution: MockVideoResolution = MockVideoResolution()
    fps: int = 30
    format: str = "YUV"


@dataclass(eq=True, frozen=True)
class MockSessionAVSettings:
    audio_input: MockSessionAudioSettings = MockSessionAudioSettings()
    audio_output: MockSessionAudioSettings = MockSessionAudioSettings()
    video_input: MockSessionVideoPublisherSettings = MockSessionVideoPublisherSettings()


@dataclass(eq=True, frozen=True)
class MockLoggingSettings:
    level: str = "INFO"


@dataclass(eq=True, frozen=True)
class MockSessionSettings:
    av: MockSessionAVSettings = MockSessionAVSettings()
    enable_migration: bool = False
    logging: MockLoggingSettings = MockLoggingSettings()


@dataclass(eq=True, frozen=True)
class MockPublisherAudioSettings:
    enable_stereo_mode: bool = True
    enable_opus_dtx: bool = False


@dataclass(eq=True, frozen=True)
class MockPublisherSettings:
    name: str = ""
    has_audio: bool = False
    has_video: bool = False
    audio_settings: MockPublisherAudioSettings = MockPublisherAudioSettings()


@dataclass(eq=True, frozen=True)
class MockVideoFrame:
    resolution: MockVideoResolution = MockVideoResolution()
    format: str = "RGB24"
    frame_buffer: bytes = b""


# Set up the mock module structure
vonage_video_mock.models.AudioData = MockAudioData
vonage_video_mock.models.Session = MockSession
vonage_video_mock.models.Stream = MockStream
vonage_video_mock.models.Publisher = MockPublisher
vonage_video_mock.models.Subscriber = MockSubscriber
vonage_video_mock.models.LoggingSettings = MockLoggingSettings
vonage_video_mock.models.SessionAVSettings = MockSessionAVSettings
vonage_video_mock.models.SessionSettings = MockSessionSettings
vonage_video_mock.models.SessionAudioSettings = MockSessionAudioSettings
vonage_video_mock.models.PublisherAudioSettings = MockPublisherAudioSettings
vonage_video_mock.models.PublisherSettings = MockPublisherSettings
vonage_video_mock.models.SessionVideoInputSettings = MockSessionVideoPublisherSettings
vonage_video_mock.models.VideoResolution = MockVideoResolution
vonage_video_mock.models.VideoFrame = MockVideoFrame

# Mock the module in sys.modules so imports work
sys.modules["vonage_video_connector"] = vonage_video_mock
sys.modules["vonage_video_connector.models"] = vonage_video_mock.models


# Now we can import the transport classes since the vonage_video module is mocked
from pipecat.transports.vonage.video_webrtc import (
    AudioProps,
    VonageClient,
    VonageClientListener,
    VonageVideoWebrtcInputTransport,
    VonageVideoWebrtcOutputTransport,
    VonageVideoWebrtcTransport,
    VonageVideoWebrtcTransportParams,
    check_audio_data,
    process_audio,
    process_audio_channels,
)


class TestVonageVideoWebrtcTransport:
    """Test cases for Vonage Video WebRTC transport classes."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.VonageClient = VonageClient
        self.VonageClientListener = VonageClientListener
        self.VonageVideoWebrtcInputTransport = VonageVideoWebrtcInputTransport
        self.VonageVideoWebrtcOutputTransport = VonageVideoWebrtcOutputTransport
        self.VonageVideoWebrtcTransport = VonageVideoWebrtcTransport
        self.VonageVideoWebrtcTransportParams = VonageVideoWebrtcTransportParams

        # Mock client instance
        self.mock_client_instance = Mock()
        vonage_video_mock.VonageVideoClient.return_value = self.mock_client_instance

        # Common test data
        self.application_id = "test-app-id"
        self.session_id = "test-session-id"
        self.token = "test-token"
        self._frame_processor_setup: Optional[FrameProcessorSetup] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _get_frame_processor_setup(self) -> FrameProcessorSetup:
        if self._frame_processor_setup is not None:
            return self._frame_processor_setup

        clock: SystemClock = SystemClock()  # type: ignore[no-untyped-call]
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        self._frame_processor_setup = FrameProcessorSetup(clock=clock, task_manager=task_manager)
        return self._frame_processor_setup

    async def _wait_for_condition(
        self,
        condition: Callable[[], bool],
        timeout: timedelta = timedelta(seconds=1),
        check_interval: timedelta = timedelta(milliseconds=10),
    ) -> None:
        """Wait for a condition to become true with timeout.

        Args:
            condition: Callable that returns True when condition is met.
            timeout: Maximum time to wait.
            check_interval: How often to check the condition.

        Raises:
            asyncio.TimeoutError: If condition is not met within timeout.
        """
        start_time = asyncio.get_event_loop().time()
        timeout_seconds = timeout.total_seconds()
        check_interval_seconds = check_interval.total_seconds()

        while not condition():
            if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                raise asyncio.TimeoutError(f"Condition not met within {timeout}")
            await asyncio.sleep(check_interval_seconds)

    def test_vonage_client_listener_defaults(self) -> None:
        """Test VonageClientListener default values."""
        listener = self.VonageClientListener()
        assert listener.on_connected is not None
        assert listener.on_disconnected is not None
        assert listener.on_error is not None
        assert listener.on_audio_in is not None
        assert listener.on_stream_received is not None
        assert listener.on_stream_dropped is not None
        assert listener.on_subscriber_connected is not None
        assert listener.on_subscriber_disconnected is not None

    def test_vonage_transport_params_defaults(self) -> None:
        """Test VonageVideoWebrtcTransportParams default values."""
        params = self.VonageVideoWebrtcTransportParams()
        assert params.publisher_name == ""
        assert params.publisher_enable_opus_dtx is False
        assert params.session_enable_migration is False

    def test_vonage_client_initialization(self) -> None:
        """Test VonageClient initialization."""
        # Reset the mock for this specific test
        vonage_video_mock.VonageVideoClient.reset_mock()

        params = self.VonageVideoWebrtcTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        assert client._application_id == self.application_id
        assert client._session_id == self.session_id
        assert client._token == self.token
        assert client._params == params
        assert client._connected is False
        assert client._connection_counter == 0
        vonage_video_mock.VonageVideoClient.assert_called_once()

        # check getting the event loop before setup raises error
        with pytest.raises(Exception) as exc_info:
            _ = client._get_event_loop()

        assert "missing task manager" in str(exc_info.value)

        # check pushing events before setup raises error
        async def mock_coro() -> None:
            pass

        mock_task = mock_coro()
        with pytest.raises(Exception) as exc_info:
            client._sdk_event_cb_to_loop(mock_task)

        mock_task.close()
        assert "missing event queue" in str(exc_info.value)

    def test_vonage_client_add_remove_listener(self) -> None:
        """Test adding and removing listeners from VonageClient."""
        params = self.VonageVideoWebrtcTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        listener = self.VonageClientListener()
        listener_id = client.add_listener(listener)

        assert isinstance(listener_id, int)
        assert listener_id in client._listeners
        assert client._listeners[listener_id] == listener

        client.remove_listener(listener_id)
        assert listener_id not in client._listeners

    def _setup_audio_ready_callback(self, client: VonageClient) -> None:
        """Helper to set up the audio ready callback."""

        def connect_side_effect(
            *_: Any, on_ready_for_audio_cb: Optional[Callable[[Any], None]] = None, **__: Any
        ) -> bool:
            assert on_ready_for_audio_cb is not None
            on_ready_for_audio_cb(vonage_video_mock.models.Session())
            return True

        self.mock_client_instance.connect = MagicMock(side_effect=connect_side_effect)

    async def _create_client(
        self,
        params: Optional[VonageVideoWebrtcTransportParams] = None,
        setup_connect_mock: bool = True,
    ) -> VonageClient:
        params = params or VonageVideoWebrtcTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)
        if setup_connect_mock:
            if params.audio_in_enabled:
                self._setup_audio_ready_callback(client)
            else:
                self.mock_client_instance.connect.return_value = True

        await client.setup(self._get_frame_processor_setup())

        return client

    async def _run_in_thread(self, callback: Callable[[], Any]) -> Any:
        """Helper to run a coroutine in a separate thread."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, callback)

    async def _wait_client_async_tasks(self, client: VonageClient) -> None:
        """Helper to wait for all async tasks in the client to complete."""
        # Wait for any pending tasks in the client's task manager
        drain_event = asyncio.Event()

        async def set_event_when_drained() -> None:
            # Wait for all queues to be joined (all tasks processed)
            if client._event_queue:
                await client._event_queue.join()
            if client._audio_queue:
                await client._audio_queue.join()
            if client._video_queue:
                await client._video_queue.join()
            drain_event.set()

        # Schedule the drain coroutine in the event loop
        asyncio.create_task(set_event_when_drained())
        await drain_event.wait()

    async def _create_output_transport(
        self, params: VonageVideoWebrtcTransportParams
    ) -> VonageVideoWebrtcOutputTransport:
        client = self.VonageClient(
            self.application_id,
            self.session_id,
            self.token,
            params,
        )
        transport = self.VonageVideoWebrtcOutputTransport(client, params)
        await transport.setup(self._get_frame_processor_setup())

        return transport

    async def _create_input_transport(
        self, params: VonageVideoWebrtcTransportParams
    ) -> VonageVideoWebrtcInputTransport:
        client = self.VonageClient(
            self.application_id,
            self.session_id,
            self.token,
            params,
        )
        transport = self.VonageVideoWebrtcInputTransport(client, params)
        await transport.setup(self._get_frame_processor_setup())

        return transport

    async def _create_transport(
        self, params: VonageVideoWebrtcTransportParams
    ) -> VonageVideoWebrtcTransport:
        transport = VonageVideoWebrtcTransport(
            self.application_id,
            self.session_id,
            self.token,
            params,
        )
        await transport.input().setup(self._get_frame_processor_setup())
        await transport.output().setup(self._get_frame_processor_setup())

        return transport

    @pytest.mark.asyncio
    async def test_vonage_client_setup_n_cleanup(self) -> None:
        """Test VonageClient setup and cleanup methods."""
        params = self.VonageVideoWebrtcTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        # Before setup, task manager and queues should be None
        assert client._task_manager is None
        assert client._event_queue is None
        assert client._event_task is None
        assert client._audio_queue is None
        assert client._audio_task is None
        assert client._video_queue is None
        assert client._video_task is None

        # Setup the client
        setup = self._get_frame_processor_setup()
        await client.setup(setup)

        # Mock connection
        self.mock_client_instance.connect.return_value = True
        client._connected = True
        client._connection_counter = 1

        # After setup, task manager and queues should be initialized
        assert client._task_manager is not None
        assert client._task_manager == setup.task_manager
        assert client._event_queue is not None
        assert client._event_task is not None
        assert client._audio_queue is not None
        assert client._audio_task is not None
        assert client._video_queue is not None
        assert client._video_task is not None

        # Test that calling setup again doesn't recreate the task manager
        old_task_manager = client._task_manager
        old_event_queue = client._event_queue
        old_event_task = client._event_task
        await client.setup(setup)
        assert client._task_manager == old_task_manager
        assert client._event_queue == old_event_queue
        assert client._event_task == old_event_task

        # Test cleanup without being connected
        await client.cleanup()

        # After cleanup, tasks should be cancelled
        assert client._event_task is None
        assert client._audio_task is None
        assert client._video_task is None

        # Verify disconnect was called
        self.mock_client_instance.disconnect.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "has_audio, has_video",
        [
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ],
    )
    async def test_vonage_client_connect_first_time(self, has_audio: bool, has_video: bool) -> None:
        """Test VonageClient connect method for first connection."""
        params = self.VonageVideoWebrtcTransportParams()

        # make changes to params depending on the configuration to check the right value
        # goes to the right destination
        params.audio_in_channels = 1 if has_video else 2
        params.audio_out_channels = 2 if has_video else 1
        params.audio_in_sample_rate = 44100 if has_video else 22050
        params.audio_out_sample_rate = 22050 if has_video else 44100
        params.session_enable_migration = has_video
        params.video_out_color_format = "YUV" if has_audio else "RGB"
        params.video_out_framerate = 30 if has_audio else 15
        params.video_out_width = 1280 if has_audio else 640
        params.video_out_height = 720 if has_audio else 480
        params.audio_in_enabled = has_audio
        params.audio_out_enabled = has_audio
        params.video_in_enabled = has_video
        params.video_out_enabled = has_video

        client = await self._create_client(params)

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True

        listener = self.VonageClientListener()
        listener.on_connected = AsyncMock()
        # only set this callback if we have audio enabled
        if has_audio:
            self._setup_audio_ready_callback(client)
        listener_id = client.add_listener(listener)
        await client.connect()

        assert isinstance(listener_id, int)
        self.mock_client_instance.connect.assert_called_once()

        # Verify connect was called with correct parameters
        call_args = self.mock_client_instance.connect.call_args
        assert call_args[1]["application_id"] == self.application_id
        assert call_args[1]["session_id"] == self.session_id
        assert call_args[1]["token"] == self.token
        assert call_args[1]["session_settings"] == MockSessionSettings(
            av=MockSessionAVSettings(
                audio_input=MockSessionAudioSettings(
                    sample_rate=params.audio_out_sample_rate,
                    number_of_channels=params.audio_out_channels,
                ),
                audio_output=MockSessionAudioSettings(
                    sample_rate=params.audio_in_sample_rate,
                    number_of_channels=params.audio_in_channels,
                ),
                video_input=MockSessionVideoPublisherSettings(
                    resolution=MockVideoResolution(
                        width=params.video_out_width,
                        height=params.video_out_height,
                    ),
                    fps=params.video_out_framerate,
                    format=self.VonageClient.vonage_image_format(params.video_out_color_format),
                ),
            ),
            enable_migration=params.session_enable_migration,
            logging=MockLoggingSettings(level="INFO"),
        )
        assert call_args[1]["on_audio_data_cb"] == client._on_session_audio_data_cb
        assert call_args[1]["on_error_cb"] == client._on_session_error_cb
        assert call_args[1]["on_connected_cb"] == client._on_session_connected_cb
        assert call_args[1]["on_disconnected_cb"] == client._on_session_disconnected_cb
        assert call_args[1]["on_ready_for_audio_cb"] is not None
        assert call_args[1]["on_stream_received_cb"] == client._on_stream_received_cb
        assert call_args[1]["on_stream_dropped_cb"] == client._on_stream_dropped_cb

        listener.on_connected.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "has_audio, has_video",
        [
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ],
    )
    async def test_vonage_client_publish_after_connect(
        self, has_audio: bool, has_video: bool
    ) -> None:
        """Test VonageClient publishes after being connected method for first connection."""
        params = self.VonageVideoWebrtcTransportParams()

        # make changes to params depending on the configuration to check the right value
        # goes to the right destination
        params.audio_in_enabled = has_audio
        params.audio_out_enabled = has_audio
        params.video_in_enabled = has_video
        params.video_out_enabled = has_video
        params.audio_out_channels = 2 if has_video else 1
        params.publisher_enable_opus_dtx = not has_video
        params.publisher_name = "test-audio" if has_audio else "test-video"

        client = await self._create_client(params)
        await client.connect()

        self.mock_client_instance.connect.assert_called_once()

        # trigger the _on_session_connected_cb to simulate being connected
        await self._run_in_thread(
            lambda: client._on_session_connected_cb(vonage_video_mock.models.Session())
        )
        await self._wait_client_async_tasks(client)

        # if no audio and no video, publish should not be called
        if not has_audio and not has_video:
            self.mock_client_instance.publish.assert_not_called()
            return

        # Verify publish was called with correct parameters
        self.mock_client_instance.publish.assert_called_once()
        call_args = self.mock_client_instance.publish.call_args
        assert call_args[1]["settings"] == MockPublisherSettings(
            name=params.publisher_name,
            audio_settings=MockPublisherAudioSettings(
                enable_stereo_mode=params.audio_out_channels == 2,
                enable_opus_dtx=params.publisher_enable_opus_dtx,
            ),
            has_audio=has_audio,
            has_video=has_video,
        )
        assert call_args[1]["on_error_cb"] == client._on_publisher_error_cb
        assert call_args[1]["on_stream_created_cb"] == client._on_publisher_stream_created_cb
        assert call_args[1]["on_stream_destroyed_cb"] == client._on_publisher_stream_destroyed_cb

    @pytest.mark.asyncio
    async def test_vonage_client_connect_already_connected(self) -> None:
        """Test VonageClient connect when already connected."""
        params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        # add some listeners, tests multiple listeners are notified, test no notifiaction after removal too
        listener1 = self.VonageClientListener()
        listener1.on_connected = AsyncMock()
        client.add_listener(listener1)
        listener2 = self.VonageClientListener()
        listener2.on_connected = AsyncMock()
        client.add_listener(listener2)
        listener3 = self.VonageClientListener()
        listener3.on_connected = AsyncMock()
        listener_id3 = client.add_listener(listener3)
        client.remove_listener(listener_id3)

        # First connection, connection is performed
        await client.connect()
        self.mock_client_instance.connect.assert_called_once()
        listener1.on_connected.assert_called_once()
        listener2.on_connected.assert_called_once()

        # Second connection, should not trigger a new connect call or raised any events
        await client.connect()
        self.mock_client_instance.connect.assert_called_once()
        listener1.on_connected.assert_called_once()
        listener2.on_connected.assert_called_once()

        # the removed listener should not have received any events
        listener3.on_connected.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_client_connect_while_disconnecting(self) -> None:
        """Test VonageClient waits for disconnect to complete before connecting."""
        params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        self.mock_client_instance.disconnect = MagicMock()

        # Simulate disconnect in progress
        disconnect_future = asyncio.get_running_loop().create_future()
        client._disconnecting_future = disconnect_future

        # Start connect task - it should block waiting for disconnect
        connect_task = asyncio.create_task(client.connect())

        # Give control to the event loop to let connect task start
        await asyncio.sleep(0.2)

        self.mock_client_instance.connect.assert_not_called()

        # Resolve the disconnect future to unblock connect
        disconnect_future.set_result(None)

        # Wait for connect to complete
        await connect_task

        self.mock_client_instance.connect.assert_called_once()

        # Verify client state
        assert client._connected is True
        assert client._connection_counter == 1

    @pytest.mark.asyncio
    async def test_vonage_client_timeout_while_connecting(self) -> None:
        """Test VonageClient handles timeout during connection."""
        params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        client = await self._create_client(params, setup_connect_mock=False)

        # Create an event that will block but can be interrupted
        stop_event = threading.Event()

        # Mock the SDK connect method to block until interrupted
        def connect_blocks_forever(*args: Any, **kwargs: Any) -> bool:
            stop_event.wait(timeout=10)  # Wait max 10 seconds but can be interrupted
            return True

        self.mock_client_instance.connect.side_effect = connect_blocks_forever

        try:
            # Patch the timeout to be very short for fast test execution
            with patch(
                "pipecat.transports.vonage.video_webrtc.VIDEO_CONNECTOR_TIMEOUT",
                timedelta(seconds=0.1),
            ):
                # Attempt to connect, should timeout
                with pytest.raises(asyncio.TimeoutError):
                    await client.connect()

                # Verify client state after timeout
                assert client._connected is False
                assert client._connection_counter == 0
                assert client._connecting_future is None
        finally:
            # Stop the blocking thread
            stop_event.set()

    @pytest.mark.asyncio
    async def test_vonage_client_concurrent_connects(self) -> None:
        """Test VonageClient concurrent connects."""
        params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        # Mock the connect method to return True and store the callback
        connecting_future: asyncio.Future[Callable[[Any], None]] = (
            asyncio.get_running_loop().create_future()
        )

        def connect_side_effect(
            *_: Any, on_ready_for_audio_cb: Optional[Callable[[Any], None]] = None, **__: Any
        ) -> bool:
            assert on_ready_for_audio_cb is not None
            connecting_future.set_result(on_ready_for_audio_cb)
            return True

        self.mock_client_instance.connect = MagicMock(side_effect=connect_side_effect)

        # create a listener
        listener = self.VonageClientListener()
        listener.on_connected = AsyncMock()
        client.add_listener(listener)

        # send two parallel connect calls and let them get stuck awaiting
        connect1_task = asyncio.create_task(client.connect())
        connect2_task = asyncio.create_task(client.connect())

        audio_ready_cb = await connecting_future

        # Now both connects are waiting on the same promise, we can set it to complete
        audio_ready_cb(vonage_video_mock.models.Session())

        # await for the connections to now complete
        await asyncio.gather(connect1_task, connect2_task)

        # SDK connect should only be called once
        self.mock_client_instance.connect.assert_called_once()
        listener.on_connected.assert_called_once()

    @pytest.mark.asyncio
    async def test_vonage_client_connect_failure(self) -> None:
        """Test VonageClient connect method when connection fails."""
        client = await self._create_client(setup_connect_mock=False)

        # Mock the connect method to return False
        self.mock_client_instance.connect.return_value = False

        with pytest.raises(Exception) as exc_info:
            await client.connect()

        assert "Could not connect to session" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_vonage_client_disconnect_before_connecting(self) -> None:
        """Test VonageClient disconnect method before connecting."""
        client = await self._create_client()

        listener = self.VonageClientListener()
        listener.on_disconnected = AsyncMock()
        client.add_listener(listener)

        await client.disconnect()

        self.mock_client_instance.disconnect.assert_not_called()
        listener.on_disconnected.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_client_disconnect(self) -> None:
        """Test VonageClient disconnect method."""
        client = await self._create_client()

        # create a listener
        listener = self.VonageClientListener()
        listener.on_disconnected = AsyncMock()
        client.add_listener(listener)

        # send two parallel connect calls and let them get stuck awaiting
        connect_promise1 = client.connect()
        connect_promise2 = client.connect()

        # await for the connections to now complete
        await asyncio.gather(connect_promise1, connect_promise2)

        # Add some items to the queues before disconnect
        assert client._event_queue is not None
        assert client._audio_queue is not None
        assert client._video_queue is not None

        async def mock_event_task() -> None:
            pass

        async def mock_audio_task() -> None:
            pass

        async def mock_video_task() -> None:
            pass

        await client._event_queue.put(mock_event_task())
        await client._audio_queue.put(mock_audio_task())
        await client._video_queue.put(mock_video_task())

        # Verify queues have items
        assert client._event_queue.qsize() == 1
        assert client._audio_queue.qsize() == 1
        assert client._video_queue.qsize() == 1

        # Mock the client's clear_media_buffers method
        self.mock_client_instance.clear_media_buffers = MagicMock()

        # send the first disconnect call, we should still be connected
        await client.disconnect()
        self.mock_client_instance.disconnect.assert_not_called()
        self.mock_client_instance.clear_media_buffers.assert_not_called()
        listener.on_disconnected.assert_not_called()

        # Queues should still have items since we didn't actually disconnect
        assert client._event_queue.qsize() == 1
        assert client._audio_queue.qsize() == 1
        assert client._video_queue.qsize() == 1

        # check the second disconnect now disconnects for real
        await client.disconnect()
        self.mock_client_instance.disconnect.assert_called_once()
        self.mock_client_instance.clear_media_buffers.assert_called_once()
        listener.on_disconnected.assert_called_once()

        # Verify queues are now empty after disconnect
        assert client._event_queue.qsize() == 0
        assert client._audio_queue.qsize() == 0
        assert client._video_queue.qsize() == 0

        # an extra disconnect should not do anything
        await client.disconnect()
        self.mock_client_instance.disconnect.assert_called_once()
        self.mock_client_instance.clear_media_buffers.assert_called_once()
        listener.on_disconnected.assert_called_once()

    @pytest.mark.asyncio
    async def test_vonage_client_disconnect_while_connecting(self) -> None:
        """Test VonageClient waits for connect to complete before disconnecting."""
        params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        client = await self._create_client(params)

        client._connected = True
        client._connection_counter = 1
        self.mock_client_instance.connect = MagicMock()

        # Simulate connect in progress
        connect_future = asyncio.get_running_loop().create_future()
        client._connecting_future = connect_future

        # Start disconnect task - it should block waiting for disconnect
        disconnect_task = asyncio.create_task(client.disconnect())

        # Give control to the event loop to let disconnect task start
        await asyncio.sleep(0.2)

        self.mock_client_instance.disconnect.assert_not_called()

        # Resolve the disconnect future to unblock connect
        connect_future.set_result(None)

        # Wait for connect to complete
        await disconnect_task

        self.mock_client_instance.disconnect.assert_called_once()

        # Verify client state
        assert client._connected is False
        assert client._connection_counter == 0

    @pytest.mark.asyncio
    async def test_vonage_client_timeout_while_disconnecting(self) -> None:
        """Test VonageClient handles timeout during disconnection."""
        params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        client = await self._create_client(params, setup_connect_mock=False)

        await client.connect()
        assert client._connection_counter == 1

        # Create an event that will block but can be interrupted
        stop_event = threading.Event()

        # Mock the SDK disconnect method to block until interrupted
        def disconnect_blocks_forever(*args: Any, **kwargs: Any) -> bool:
            stop_event.wait(timeout=10)  # Wait max 10 seconds but can be interrupted
            return True

        self.mock_client_instance.disconnect.side_effect = disconnect_blocks_forever
        try:
            # Patch the timeout to be very short for fast test execution
            with patch(
                "pipecat.transports.vonage.video_webrtc.VIDEO_CONNECTOR_TIMEOUT",
                timedelta(seconds=0.1),
            ):
                # Attempt to connect, should timeout
                with pytest.raises(asyncio.TimeoutError):
                    await client.disconnect()

                # Verify client state after timeout
                assert client._connected is True
                assert client._connection_counter == 1
                assert client._disconnecting_future is None
        finally:
            # Stop the blocking thread
            stop_event.set()

    @pytest.mark.asyncio
    async def test_vonage_client_clear_media_buffers(self) -> None:
        """Test VonageClient clear_media_buffers method."""
        params = self.VonageVideoWebrtcTransportParams(
            audio_out_channels=2, audio_out_sample_rate=48000
        )
        client = await self._create_client(params)

        # Add some items to the audio and video queues
        assert client._audio_queue is not None
        assert client._video_queue is not None

        # Create mock coroutines to add to queues
        async def mock_audio_task() -> None:
            pass

        async def mock_video_task() -> None:
            pass

        # Put some items in the queues
        await client._audio_queue.put(mock_audio_task())
        await client._audio_queue.put(mock_audio_task())
        await client._video_queue.put(mock_video_task())

        # Verify queues have items
        assert client._audio_queue.qsize() == 2
        assert client._video_queue.qsize() == 1

        # Mock the client's clear_media_buffers method
        self.mock_client_instance.clear_media_buffers = MagicMock()

        # Clear the buffers
        client.clear_media_buffers()

        # Verify queues are now empty
        assert client._audio_queue.qsize() == 0
        assert client._video_queue.qsize() == 0

        # Verify the SDK client's clear_media_buffers was called
        self.mock_client_instance.clear_media_buffers.assert_called_once()

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.EVENT_QUEUE_MAXSIZE", 1)
    async def test_vonage_client_sdk_cb_to_loop_full_queue(self) -> None:
        """Test VonageClient SDK callback to loop filling up the queue."""
        params = self.VonageVideoWebrtcTransportParams()
        client = await self._create_client(params)

        # Ensure the loop thread ID is set
        assert client._event_queue is not None
        assert client._loop_thread_id == threading.get_ident()

        # Create a mock coroutine to queue
        async def mock_task() -> None:
            pass

        # Fill queue to max size
        for _ in range(client._event_queue.maxsize):
            await client._event_queue.put(mock_task())

        # Queue should be full
        assert client._event_queue.qsize() == client._event_queue.maxsize

        # This should log an error and drop the event
        async_task = mock_task()
        client._sdk_cb_to_loop("test_event", client._event_queue, async_task)

        # Queue should still be full (no new item added)
        assert client._event_queue.qsize() == client._event_queue.maxsize
        # check the coroutine was closed and hence dropped
        assert inspect.getcoroutinestate(async_task) == "CORO_CLOSED"

        # Clean up the coroutine
        task = await client._event_queue.get()
        task.close()
        client._event_queue.task_done()

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_client_get_audio_with_resampling(self, mock_resampler: MagicMock) -> None:
        """Test VonageClient get_audio method."""
        # Return resampled stereo data
        resampled_data = b"\x07\x06\x05\x04\x03\x02\x01\x00"
        mock_resampler_instance = Mock()
        mock_resampler_instance.resample = AsyncMock(return_value=resampled_data)
        mock_resampler.return_value = mock_resampler_instance

        params = self.VonageVideoWebrtcTransportParams(
            audio_in_channels=1,
            audio_in_sample_rate=48000,
            audio_in_enabled=True,
        )
        client = await self._create_client(params)
        listener = self.VonageClientListener()
        on_audio_in_mock = AsyncMock()
        listener.on_audio_in = on_audio_in_mock
        client.add_listener(listener)

        await client.connect()

        mock_audio_data = vonage_video_mock.models.AudioData(
            sample_buffer=memoryview(b"\x00\x01\x02\x03\x04\x05\x06\x07"),
            number_of_frames=4,
            number_of_channels=1,
            sample_rate=16000,
        )

        session = vonage_video_mock.models.Session(id="test_session")
        client._on_session_audio_data_cb(session, mock_audio_data)
        await self._wait_for_condition(lambda: on_audio_in_mock.call_count > 0)

        listener.on_audio_in.assert_called_once_with(session, ANY)
        frame = listener.on_audio_in.call_args[0][1]
        assert frame.audio == resampled_data
        assert frame.num_frames == 4
        assert frame.sample_rate == 48000
        assert frame.num_channels == 1

    @pytest.mark.asyncio
    async def test_vonage_client_write_audio(self) -> None:
        """Test VonageClient write_audio method."""
        params = self.VonageVideoWebrtcTransportParams(
            audio_out_channels=2, audio_out_sample_rate=48000
        )
        client = await self._create_client(params)

        # Create mock audio data
        audio_data = OutputAudioRawFrame(
            audio=b"\x00\x01\x02\x03\x04\x05\x06\x07",
            sample_rate=48000,
            num_channels=2,
        )  # 4 frames of 2-channel 16-bit audio

        await client.write_audio(audio_data)

        self.mock_client_instance.add_audio.assert_called_once()
        call_args = self.mock_client_instance.add_audio.call_args[0][0]
        assert call_args.sample_buffer.tobytes() == audio_data.audio
        assert call_args.number_of_frames == 2  # 8 bytes / (2 channels * 2 bytes)
        assert call_args.number_of_channels == 2
        assert call_args.sample_rate == 48000

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_client_write_audio_with_resampling(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test VonageClient write_audio method."""
        # Return resampled stereo data
        resampled_data = b"\x07\x06\x05\x04\x03\x02\x01\x00"
        mock_resampler_instance = Mock()
        mock_resampler_instance.resample = AsyncMock(return_value=resampled_data)
        mock_resampler.return_value = mock_resampler_instance

        params = self.VonageVideoWebrtcTransportParams(
            audio_out_channels=1, audio_out_sample_rate=16000
        )
        client = await self._create_client(params)

        # Create mock audio data
        audio_data = OutputAudioRawFrame(
            audio=b"\x00\x01\x02\x03\x04\x05\x06\x07",
            sample_rate=48000,
            num_channels=1,
        )  # 4 frames of 1-channel 16-bit audio

        await client.write_audio(audio_data)

        self.mock_client_instance.add_audio.assert_called_once()
        call_args = self.mock_client_instance.add_audio.call_args[0][0]
        assert call_args.sample_buffer.tobytes() == resampled_data
        assert call_args.number_of_frames == 4  # 8 bytes / (1 channel * 2 bytes)
        assert call_args.number_of_channels == 1
        assert call_args.sample_rate == 16000

    @pytest.mark.asyncio
    async def test_vonage_client_write_video(self) -> None:
        """Test VonageClient write_video method."""
        params = self.VonageVideoWebrtcTransportParams(
            video_out_width=640,
            video_out_height=480,
            video_out_color_format="RGB",
        )
        client = await self._create_client(params)

        # Create a test RGB image (640x480, 3 channels)
        width, height = 640, 480
        # Create RGB data: simple gradient pattern
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 100  # R channel
        rgb_image[:, :, 1] = 150  # G channel
        rgb_image[:, :, 2] = 200  # B channel

        rgb_bytes = rgb_image.tobytes()

        # Create OutputImageRawFrame
        frame = OutputImageRawFrame(image=rgb_bytes, size=(width, height), format="RGB")

        # Mock the add_video method
        self.mock_client_instance.add_video = MagicMock(return_value=True)

        result = await client.write_video(frame)

        # Verify add_video was called
        assert result is True
        self.mock_client_instance.add_video.assert_called_once()

        # Get the VideoFrame argument
        call_args = self.mock_client_instance.add_video.call_args[0][0]

        # Verify the resolution
        assert call_args.resolution.width == width
        assert call_args.resolution.height == height

        # Verify the format
        assert call_args.format == "RGB24"

        # Verify BGR conversion happened correctly
        # Convert back from the buffer to verify
        bgr_buffer = bytes(call_args.frame_buffer)
        bgr_image = np.frombuffer(bgr_buffer, dtype=np.uint8).reshape(height, width, 3)

        # Check that RGB was converted to BGR (channels swapped)
        assert bgr_image[0, 0, 0] == 200  # B channel (was R=200 in RGB)
        assert bgr_image[0, 0, 1] == 150  # G channel (unchanged)
        assert bgr_image[0, 0, 2] == 100  # R channel (was B=100 in RGB)

    @pytest.mark.asyncio
    async def test_vonage_client_events(self) -> None:
        """Test VonageClient events"""
        params = self.VonageVideoWebrtcTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=48000,
            audio_in_channels=2,
        )
        client = await self._create_client(params)

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True
        self._setup_audio_ready_callback(client)

        # create a listener
        listener = self.VonageClientListener()
        on_error_mock = AsyncMock()
        listener.on_error = on_error_mock
        on_audio_in_mock = AsyncMock()
        listener.on_audio_in = on_audio_in_mock
        on_stream_received_mock = AsyncMock()
        listener.on_stream_received = on_stream_received_mock
        on_stream_dropped_mock = AsyncMock()
        listener.on_stream_dropped = on_stream_dropped_mock
        on_subscriber_connected_mock = AsyncMock()
        listener.on_subscriber_connected = on_subscriber_connected_mock
        on_subscriber_disconnected_mock = AsyncMock()
        listener.on_subscriber_disconnected = on_subscriber_disconnected_mock

        client.add_listener(listener)

        # connect
        await client.connect()

        # Test _on_session_error_cb triggers on_error
        session = vonage_video_mock.models.Session(id="test_session")
        error_description = "Test error description"
        error_code = 500

        client._on_session_error_cb(session, error_description, error_code)
        await self._wait_for_condition(lambda: on_error_mock.call_count > 0)

        listener.on_error.assert_called_once_with(session, error_description, error_code)
        listener.on_error.reset_mock()

        # Test _on_session_audio_data_cb triggers on_audio_in
        audio_buffer = np.array([100, 200, 300, 400], dtype=np.int16)
        mock_audio_data = vonage_video_mock.models.AudioData(
            sample_buffer=memoryview(audio_buffer),
            number_of_frames=2,
            number_of_channels=2,
            sample_rate=48000,
        )

        client._on_session_audio_data_cb(session, mock_audio_data)
        await self._wait_for_condition(lambda: on_audio_in_mock.call_count > 0)

        listener.on_audio_in.assert_called_once_with(session, ANY)
        frame = listener.on_audio_in.call_args[0][1]
        assert frame.audio == audio_buffer.tobytes()
        assert frame.sample_rate == 48000
        assert frame.num_channels == 2
        listener.on_audio_in.reset_mock()

        # Test _on_stream_received_cb triggers on_stream_received
        stream = vonage_video_mock.models.Stream(id="test_stream")
        self.mock_client_instance.subscribe = MagicMock()

        client._on_stream_received_cb(session, stream)
        await self._wait_for_condition(lambda: on_stream_received_mock.call_count > 0)

        listener.on_stream_received.assert_called_once_with(session, stream)
        self.mock_client_instance.subscribe.assert_called_once_with(
            stream,
            on_error_cb=client._on_subscriber_error_cb,
            on_connected_cb=client._on_subscriber_connected_cb,
            on_disconnected_cb=client._on_subscriber_disconnected_cb,
        )
        listener.on_stream_received.reset_mock()

        # Test _on_stream_dropped_cb triggers on_stream_dropped
        self.mock_client_instance.unsubscribe = MagicMock()

        client._on_stream_dropped_cb(session, stream)
        await self._wait_for_condition(lambda: on_stream_dropped_mock.call_count > 0)

        listener.on_stream_dropped.assert_called_once_with(session, stream)
        self.mock_client_instance.unsubscribe.assert_called_once_with(stream)
        listener.on_stream_dropped.reset_mock()

        # Test _on_subscriber_connected_cb triggers on_subscriber_connected
        subscriber_stream = vonage_video_mock.models.Stream(id="subscriber_stream")
        subscriber = vonage_video_mock.models.Subscriber(stream=subscriber_stream)

        client._on_subscriber_connected_cb(subscriber)
        await self._wait_for_condition(lambda: on_subscriber_connected_mock.call_count > 0)

        listener.on_subscriber_connected.assert_called_once_with(subscriber)
        listener.on_subscriber_connected.reset_mock()

        # Test _on_subscriber_disconnected_cb triggers on_subscriber_disconnected
        client._on_subscriber_disconnected_cb(subscriber)
        await self._wait_for_condition(lambda: on_subscriber_disconnected_mock.call_count > 0)

        listener.on_subscriber_disconnected.assert_called_once_with(subscriber)
        listener.on_subscriber_disconnected.reset_mock()

        # Test error callbacks are logged but don't trigger listener events
        # (these are internal error callbacks, not session errors)
        publisher_stream = vonage_video_mock.models.Stream(id="publisher_stream")
        publisher = vonage_video_mock.models.Publisher(stream=publisher_stream)

        # These should not raise exceptions
        client._on_publisher_error_cb(publisher, "publisher error", 400)
        client._on_subscriber_error_cb(subscriber, "subscriber error", 401)

    @pytest.mark.asyncio
    async def test_vonage_input_transport_initialization(self) -> None:
        """Test VonageVideoWebrtcInputTransport initialization."""
        params = self.VonageVideoWebrtcTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)

        assert transport._client == client
        assert transport._initialized is False

    @pytest.mark.asyncio
    async def test_vonage_input_transport_start(self) -> None:
        """Test VonageVideoWebrtcInputTransport start method."""
        params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)
        transport = self.VonageVideoWebrtcInputTransport(client, params)

        # Mock the client connect method
        with (
            patch.object(client, "connect", AsyncMock(return_value=1)) as client_connect_mock,
            patch.object(transport, "set_transport_ready", AsyncMock()) as set_transport_ready_mock,
        ):
            start_frame = StartFrame()
            await transport.start(start_frame)

            assert transport._initialized is True
            assert transport._connected is True
            client_connect_mock.assert_called_once()
            set_transport_ready_mock.assert_called_once_with(start_frame)

    @pytest.mark.asyncio
    async def test_vonage_input_transport_stop(self) -> None:
        """Test VonageVideoWebrtcInputTransport stop method."""
        params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)
        transport = self.VonageVideoWebrtcInputTransport(client, params)
        transport._listener_id = 1
        transport._connected = True

        with (
            patch.object(client, "disconnect", AsyncMock()) as client_disconnect_mock,
            patch.object(client, "remove_listener", MagicMock()) as remove_listener_mock,
        ):
            end_frame = EndFrame()
            await transport.stop(end_frame)

            client_disconnect_mock.assert_called_once()
            remove_listener_mock.assert_called_once_with(1)
            assert not transport._connected

    @pytest.mark.asyncio
    async def test_vonage_input_transport_cancel(self) -> None:
        """Test VonageVideoWebrtcInputTransport cancel method."""
        params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport = self.VonageVideoWebrtcInputTransport(client, params)
        transport._listener_id = 1
        transport._connected = True

        # Mock the client disconnect method
        with (
            patch.object(client, "disconnect", AsyncMock()) as client_disconnect_mock,
            patch.object(client, "remove_listener", MagicMock()) as remove_listener_mock,
        ):
            cancel_frame = CancelFrame()
            await transport.cancel(cancel_frame)

            client_disconnect_mock.assert_called_once()
            remove_listener_mock.assert_called_once_with(1)
            assert not transport._connected

    @pytest.mark.asyncio
    async def test_vonage_output_transport_initialization(self) -> None:
        """Test VonageVideoWebrtcOutputTransport initialization."""
        params = self.VonageVideoWebrtcTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        transport = self.VonageVideoWebrtcOutputTransport(client, transport_params)

        assert transport._client == client
        assert transport._initialized is False

    @pytest.mark.asyncio
    async def test_vonage_output_transport_start(self) -> None:
        """Test VonageVideoWebrtcOutputTransport start method."""
        params = self.VonageVideoWebrtcTransportParams()
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        transport_params = self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        transport = self.VonageVideoWebrtcOutputTransport(client, transport_params)

        with (
            patch.object(client, "connect", AsyncMock(return_value=1)) as client_connect_mock,
            patch.object(transport, "set_transport_ready", AsyncMock()) as set_transport_ready_mock,
        ):
            start_frame = StartFrame()
            await transport.start(start_frame)

            assert transport._initialized is True
            client_connect_mock.assert_called_once()
            set_transport_ready_mock.assert_called_once_with(start_frame)

    @pytest.mark.asyncio
    async def test_vonage_output_transport_write_audio_frame(self) -> None:
        """Test VonageVideoWebrtcOutputTransport write_audio_frame method."""

        params = self.VonageVideoWebrtcTransportParams(
            audio_out_sample_rate=48000, audio_out_channels=2, audio_out_enabled=True
        )
        client = self.VonageClient(self.application_id, self.session_id, self.token, params)

        with patch.object(client, "write_audio", AsyncMock()) as client_write_audio_mock:
            transport_params = self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
            transport = self.VonageVideoWebrtcOutputTransport(client, transport_params)
            transport._connected = True

            # Create a mock audio frame
            audio_frame = OutputAudioRawFrame(
                audio=b"\x00\x01\x02\x03", sample_rate=16000, num_channels=1
            )

            await transport.write_audio_frame(audio_frame)

            # Verify audio was written to client
            client_write_audio_mock.assert_called_once_with(audio_frame)

    @pytest.mark.asyncio
    async def test_vonage_output_transport_write_video_frame_not_connected(self) -> None:
        """Test VonageVideoWebrtcOutputTransport write_video_frame method."""
        transport = await self._create_output_transport(
            params=self.VonageVideoWebrtcTransportParams(video_out_enabled=True)
        )
        client = transport._client

        # Create a test video frame
        width, height = 640, 480
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 100
        rgb_image[:, :, 1] = 150
        rgb_image[:, :, 2] = 200

        video_frame = OutputImageRawFrame(
            image=rgb_image.tobytes(), size=(width, height), format="RGB"
        )

        with patch.object(client, "write_video", AsyncMock(return_value=True)) as write_video_mock:
            await transport.stop(EndFrame())
            result = await transport.write_video_frame(video_frame)

            # Should return False when not connected
            assert result is False
            write_video_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_output_transport_write_video_frame_connected(self) -> None:
        """Test VonageVideoWebrtcOutputTransport write_video_frame method when connected."""
        transport = await self._create_output_transport(
            params=self.VonageVideoWebrtcTransportParams(
                video_out_enabled=True,
                video_out_width=640,
                video_out_height=480,
                video_out_color_format="RGB",
            )
        )
        client = transport._client

        # Create a test video frame
        width, height = 640, 480
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 100
        rgb_image[:, :, 1] = 150
        rgb_image[:, :, 2] = 200

        video_frame = OutputImageRawFrame(
            image=rgb_image.tobytes(), size=(width, height), format="RGB"
        )

        with patch.object(client, "write_video", AsyncMock(return_value=True)) as write_video_mock:
            transport._connected = True
            result = await transport.write_video_frame(video_frame)

            # Should return True and call write_video when connected
            assert result is True
            write_video_mock.assert_called_once_with(video_frame)

    @pytest.mark.asyncio
    async def test_vonage_output_transport_write_video_frame_invalid_size(self) -> None:
        """Test VonageVideoWebrtcOutputTransport write_video_frame with invalid frame size."""
        transport = await self._create_output_transport(
            params=self.VonageVideoWebrtcTransportParams(
                video_out_enabled=True,
                video_out_width=640,
                video_out_height=480,
                video_out_color_format="RGB",
            )
        )

        # Create a video frame with incorrect size
        width, height = 320, 240  # Different from expected 640x480
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        video_frame = OutputImageRawFrame(
            image=rgb_image.tobytes(), size=(width, height), format="RGB"
        )

        transport._connected = True
        result = await transport.write_video_frame(video_frame)

        # Should return False for invalid size
        assert result is False

    @pytest.mark.asyncio
    async def test_vonage_output_transport_write_video_frame_invalid_format(self) -> None:
        """Test VonageVideoWebrtcOutputTransport write_video_frame with invalid color format."""
        transport = await self._create_output_transport(
            params=self.VonageVideoWebrtcTransportParams(
                video_out_enabled=True,
                video_out_width=640,
                video_out_height=480,
                video_out_color_format="YUV",
            )
        )

        # Create a video frame with incorrect size
        width, height = 320, 240  # Different from expected 640x480
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        video_frame = OutputImageRawFrame(
            image=rgb_image.tobytes(), size=(width, height), format="RGB"
        )

        transport._connected = True
        result = await transport.write_video_frame(video_frame)

        # Should return False for invalid size
        assert result is False

    @pytest.mark.asyncio
    async def test_vonage_output_transport_process_frame_with_interruption(self) -> None:
        """Test VonageVideoWebrtcOutputTransport process_frame method with InterruptionFrame."""
        transport = await self._create_output_transport(
            params=self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        )
        client = transport._client

        with (
            patch.object(client, "clear_media_buffers") as clear_buffers_mock,
            patch.object(client, "connect", AsyncMock()),
        ):
            await transport.start(StartFrame())
            interruption_frame = InterruptionFrame()
            await transport.process_frame(interruption_frame, FrameDirection.DOWNSTREAM)

            # Verify clear_media_buffers was called
            clear_buffers_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_vonage_output_transport_process_frame_without_interruption(self) -> None:
        """Test VonageVideoWebrtcOutputTransport process_frame method with non-interruption frame."""
        transport = await self._create_output_transport(
            params=self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        )
        client = transport._client

        with patch.object(client, "clear_media_buffers") as clear_buffers_mock:
            audio_frame = OutputAudioRawFrame(
                audio=b"\x00\x01\x02\x03", sample_rate=16000, num_channels=1
            )
            await transport.process_frame(audio_frame, FrameDirection.DOWNSTREAM)

            # Verify clear_media_buffers was NOT called for non-interruption frames
            clear_buffers_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_output_transport_process_frame_when_not_connected(self) -> None:
        """Test VonageVideoWebrtcOutputTransport process_frame method when not connected."""
        transport = await self._create_output_transport(
            params=self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        )
        await transport.stop(EndFrame())  # Ensure transport is not connected
        client = transport._client

        with patch.object(client, "clear_media_buffers") as clear_buffers_mock:
            interruption_frame = InterruptionFrame()
            await transport.process_frame(interruption_frame, FrameDirection.DOWNSTREAM)

            # Verify clear_media_buffers was NOT called when not connected
            clear_buffers_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_transport_initialization(self) -> None:
        """Test VonageVideoWebrtcTransport initialization."""
        params = self.VonageVideoWebrtcTransportParams(
            audio_out_sample_rate=48000,
            audio_out_channels=2,
            audio_out_enabled=True,
            session_enable_migration=True,
            publisher_name="test-publisher",
            publisher_enable_opus_dtx=True,
        )
        transport = await self._create_transport(params=params)

        assert transport._client is not None
        assert transport._one_stream_received is False

        # Verify vonage client was initialized with correct parameters
        client_params = transport._client._params
        assert client_params.audio_out_sample_rate == 48000
        assert client_params.audio_out_channels == 2
        assert client_params.session_enable_migration is True

    @pytest.mark.asyncio
    async def test_vonage_transport_input_output_methods(self) -> None:
        """Test VonageVideoWebrtcTransport input and output methods."""
        params = self.VonageVideoWebrtcTransportParams()
        transport = self.VonageVideoWebrtcTransport(
            self.application_id, self.session_id, self.token, params
        )

        # Test input method
        input_transport = transport.input()
        assert isinstance(input_transport, self.VonageVideoWebrtcInputTransport)

        # Test output method
        output_transport = transport.output()
        assert isinstance(output_transport, self.VonageVideoWebrtcOutputTransport)

        # Verify they return the same instances on subsequent calls
        assert transport.input() is input_transport
        assert transport.output() is output_transport

    @pytest.mark.asyncio
    async def test_vonage_input_audio_callback(self) -> None:
        """Test audio input callback processing."""

        params = self.VonageVideoWebrtcTransportParams(
            audio_in_enabled=True,
        )
        transport = await self._create_input_transport(params)
        client = transport._client

        with (
            patch.object(transport, "push_audio_frame", AsyncMock()) as mock_push_audio_frame,
            patch.object(client, "connect", AsyncMock(return_value=1)),
        ):
            start_frame = StartFrame()
            await transport.start(start_frame)

            # Create mock audio data
            audio_buffer = np.array([100, 200, 300, 400], dtype=np.int16)
            audio_frame = InputAudioRawFrame(
                audio=audio_buffer.tobytes(), sample_rate=48000, num_channels=2
            )

            # Call the audio callback
            await transport._audio_in_cb(vonage_video_mock.models.Session(), audio_frame)

            mock_push_audio_frame.assert_called_once_with(audio_frame)

    @pytest.mark.asyncio
    async def test_vonage_transport_event_handlers(self) -> None:
        """Test VonageVideoWebrtcTransport event handlers."""
        params = self.VonageVideoWebrtcTransportParams()
        transport = await self._create_transport(params)

        with patch.object(
            transport, "_call_event_handler", new_callable=AsyncMock
        ) as mock_call_event_handler:
            # Test session events
            mock_session = Mock()
            mock_session.id = "session-123"

            await transport._on_connected(mock_session)
            mock_call_event_handler.assert_called_with("on_joined", {"sessionId": "session-123"})

            await transport._on_disconnected(mock_session)
            mock_call_event_handler.assert_called_with("on_left")

            await transport._on_error(mock_session, "test error", 500)
            mock_call_event_handler.assert_called_with("on_error", "test error")

            # Test stream events
            mock_stream = Mock()
            mock_stream.id = "stream-456"

            await transport._on_stream_received(mock_session, mock_stream)
            # Should call both first participant and participant joined events
            expected_calls = [
                call(
                    "on_first_participant_joined",
                    {"sessionId": "session-123", "streamId": "stream-456"},
                ),
                call(
                    "on_participant_joined", {"sessionId": "session-123", "streamId": "stream-456"}
                ),
            ]
            mock_call_event_handler.assert_has_calls(expected_calls)

            await transport._on_stream_dropped(mock_session, mock_stream)
            mock_call_event_handler.assert_called_with(
                "on_participant_left", {"sessionId": "session-123", "streamId": "stream-456"}
            )

            # Test subscriber events
            mock_subscriber = Mock()
            mock_subscriber.stream.id = "subscriber-789"

            await transport._on_subscriber_connected(mock_subscriber)
            mock_call_event_handler.assert_called_with(
                "on_client_connected", {"subscriberId": "subscriber-789"}
            )

            await transport._on_subscriber_disconnected(mock_subscriber)
            mock_call_event_handler.assert_called_with(
                "on_client_disconnected", {"subscriberId": "subscriber-789"}
            )

    @pytest.mark.asyncio
    async def test_vonage_transport_first_participant_flag(self) -> None:
        """Test that first participant event is only called once."""
        params = self.VonageVideoWebrtcTransportParams()
        transport = await self._create_transport(params)

        with patch.object(
            transport, "_call_event_handler", new_callable=AsyncMock
        ) as mock_call_event_handler:
            mock_session = Mock()
            mock_session.id = "session-123"
            mock_stream1 = Mock()
            mock_stream1.id = "stream-456"
            mock_stream2 = Mock()
            mock_stream2.id = "stream-789"

            # First stream should trigger first participant event
            await transport._on_stream_received(mock_session, mock_stream1)
            assert transport._one_stream_received is True

            # Reset mock to check second stream
            mock_call_event_handler.reset_mock()

            # Second stream should not trigger first participant event
            await transport._on_stream_received(mock_session, mock_stream2)
            mock_call_event_handler.assert_called_once_with(
                "on_participant_joined", {"sessionId": "session-123", "streamId": "stream-789"}
            )


class TestAudioNormalization:
    """Test cases for audio normalization functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.AudioProps = AudioProps
        self.process_audio_channels = process_audio_channels
        self.process_audio = process_audio
        self.check_audio_data = check_audio_data

    def test_audio_props_creation(self) -> None:
        """Test AudioProps dataclass creation."""
        props = self.AudioProps(sample_rate=48000, is_stereo=True)
        assert props.sample_rate == 48000
        assert props.is_stereo is True

        props_mono = self.AudioProps(sample_rate=16000, is_stereo=False)
        assert props_mono.sample_rate == 16000
        assert props_mono.is_stereo is False

    def test_process_audio_channels_mono_to_stereo(self) -> None:
        """Test converting mono audio to stereo."""
        # Create mono audio (4 samples)
        mono_audio = np.array([100, 200, 300, 400], dtype=np.int16)

        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=48000, is_stereo=True)

        result = self.process_audio_channels(mono_audio, current, target)

        # Should duplicate each sample
        expected = np.array([100, 100, 200, 200, 300, 300, 400, 400], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_process_audio_channels_stereo_to_mono(self) -> None:
        """Test converting stereo audio to mono."""
        # Create stereo audio (2 frames, 4 samples total)
        stereo_audio = np.array([100, 200, 300, 400], dtype=np.int16)

        current = self.AudioProps(sample_rate=48000, is_stereo=True)
        target = self.AudioProps(sample_rate=48000, is_stereo=False)

        result = self.process_audio_channels(stereo_audio, current, target)

        # Should average each stereo pair: (100+200)/2=150, (300+400)/2=350
        expected = np.array([150, 350], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_process_audio_channels_same_format(self) -> None:
        """Test when source and target have the same channel format."""
        audio = np.array([100, 200, 300, 400], dtype=np.int16)

        # Test mono to mono
        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=48000, is_stereo=False)
        result = self.process_audio_channels(audio, current, target)
        np.testing.assert_array_equal(result, audio)

        # Test stereo to stereo
        current = self.AudioProps(sample_rate=48000, is_stereo=True)
        target = self.AudioProps(sample_rate=48000, is_stereo=True)
        result = self.process_audio_channels(audio, current, target)
        np.testing.assert_array_equal(result, audio)

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_process_audio_same_sample_rate(self, mock_resampler: MagicMock) -> None:
        """Test process_audio when sample rates are the same."""
        mock_resampler_instance = Mock()
        mock_resampler.return_value = mock_resampler_instance

        audio = np.array([100, 200, 300, 400], dtype=np.int16)
        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=48000, is_stereo=True)

        result = await self.process_audio(mock_resampler_instance, audio, current, target)

        # Should only do channel conversion, no resampling
        expected = np.array([100, 100, 200, 200, 300, 300, 400, 400], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

        # Resampler should not be called
        mock_resampler_instance.resample.assert_not_called()

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_process_audio_different_sample_rate_mono(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test process_audio with different sample rates (mono)."""
        mock_resampler_instance = Mock()
        mock_resampler_instance.resample = AsyncMock(
            return_value=b"\x64\x00\xc8\x00"
        )  # 100, 200 in bytes
        mock_resampler.return_value = mock_resampler_instance

        audio = np.array([150, 250, 350, 450], dtype=np.int16)
        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=16000, is_stereo=False)

        result = await self.process_audio(mock_resampler_instance, audio, current, target)

        # Should resample the audio
        expected = np.array([100, 200], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

        # Resampler should be called with correct parameters
        mock_resampler_instance.resample.assert_called_once_with(audio.tobytes(), 48000, 16000)

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_process_audio_different_sample_rate_stereo_to_mono(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test process_audio with different sample rates and channel conversion."""
        mock_resampler_instance = Mock()
        # Return resampled mono data
        mock_resampler_instance.resample = AsyncMock(
            return_value=b"\x64\x00\xc8\x00"
        )  # 100, 200 in bytes
        mock_resampler.return_value = mock_resampler_instance

        # Stereo audio: 2 frames with left/right channels
        audio = np.array([100, 200, 300, 400], dtype=np.int16)  # L1=100, R1=200, L2=300, R2=400
        current = self.AudioProps(sample_rate=48000, is_stereo=True)
        target = self.AudioProps(sample_rate=16000, is_stereo=False)

        result = await self.process_audio(mock_resampler_instance, audio, current, target)

        # Should convert to mono first, then resample
        expected = np.array([100, 200], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

        # Resampler should be called with mono audio
        expected_mono = np.array([150, 350], dtype=np.int16)  # (100+200)/2, (300+400)/2
        mock_resampler_instance.resample.assert_called_once_with(
            expected_mono.tobytes(), 48000, 16000
        )

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_process_audio_different_sample_rate_mono_to_stereo(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test process_audio with different sample rates converting mono to stereo."""
        mock_resampler_instance = Mock()
        # Return resampled mono data
        mock_resampler_instance.resample = AsyncMock(
            return_value=b"\x64\x00\xc8\x00"
        )  # 100, 200 in bytes
        mock_resampler.return_value = mock_resampler_instance

        audio = np.array([150, 250], dtype=np.int16)
        current = self.AudioProps(sample_rate=48000, is_stereo=False)
        target = self.AudioProps(sample_rate=16000, is_stereo=True)

        result = await self.process_audio(mock_resampler_instance, audio, current, target)

        # Should resample first (mono), then convert to stereo
        expected = np.array([100, 100, 200, 200], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

        # Resampler should be called with mono audio
        mock_resampler_instance.resample.assert_called_once_with(audio.tobytes(), 48000, 16000)

    def test_check_audio_data_valid_mono_bytes(self) -> None:
        """Test check_audio_data with valid mono audio as bytes."""
        # 4 frames of mono 16-bit audio (8 bytes total)
        buffer = b"\x00\x01\x02\x03\x04\x05\x06\x07"

        # Should not raise any exception
        self.check_audio_data(buffer, 4, 1)

    def test_check_audio_data_valid_stereo_bytes(self) -> None:
        """Test check_audio_data with valid stereo audio as bytes."""
        # 2 frames of stereo 16-bit audio (8 bytes total)
        buffer = b"\x00\x01\x02\x03\x04\x05\x06\x07"

        # Should not raise any exception
        self.check_audio_data(buffer, 2, 2)

    def test_check_audio_data_valid_memoryview(self) -> None:
        """Test check_audio_data with valid audio as memoryview."""
        # Create int16 memoryview (2 bytes per sample)
        array = np.array([100, 200, 300, 400], dtype=np.int16)
        buffer = memoryview(array)

        # Should not raise any exception
        self.check_audio_data(buffer, 4, 1)  # 4 mono frames
        self.check_audio_data(buffer, 2, 2)  # 2 stereo frames

    def test_check_audio_data_invalid_channels(self) -> None:
        """Test check_audio_data with invalid number of channels."""
        buffer = b"\x00\x01\x02\x03"

        # Should raise ValueError for invalid channel counts
        with pytest.raises(ValueError) as exc_info:
            self.check_audio_data(buffer, 2, 3)  # 3 channels not supported
        assert "mono or stereo" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            self.check_audio_data(buffer, 2, 0)  # 0 channels not supported
        assert "mono or stereo" in str(exc_info.value)

    def test_check_audio_data_invalid_bit_depth_bytes(self) -> None:
        """Test check_audio_data with invalid bit depth using bytes."""
        # 2 frames of mono audio with 1 byte per sample (8-bit)
        buffer = b"\x00\x01"

        with pytest.raises(ValueError) as exc_info:
            self.check_audio_data(buffer, 2, 1)
        assert "16 bit PCM" in str(exc_info.value)
        assert "got 8 bit" in str(exc_info.value)

    def test_check_audio_data_invalid_bit_depth_memoryview(self) -> None:
        """Test check_audio_data with invalid bit depth using memoryview."""
        # Create uint8 memoryview (1 byte per sample)
        array = np.array([100, 200], dtype=np.uint8)
        buffer = memoryview(array)

        with pytest.raises(ValueError) as exc_info:
            self.check_audio_data(buffer, 2, 1)
        assert "16 bit PCM" in str(exc_info.value)
        assert "got 8 bit" in str(exc_info.value)

    def test_check_audio_data_buffer_size_mismatch(self) -> None:
        """Test check_audio_data with buffer size that doesn't match expected size."""
        # 3 bytes total, but expecting 2 frames of mono 16-bit (should be 4 bytes)
        buffer = b"\x00\x01\x02"

        with pytest.raises(ValueError) as exc_info:
            self.check_audio_data(buffer, 2, 1)
        # Should detect that 3 bytes / (2 frames * 1 channel) = 1.5 bytes per sample
        # which gets truncated to 1 byte per sample = 8 bit
        assert "16 bit PCM" in str(exc_info.value)
