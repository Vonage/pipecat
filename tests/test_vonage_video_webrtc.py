# SPDX-License-Identifier: BSD 2-Clause License

import asyncio
import sys
import unittest
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import timedelta
from http import client
from typing import Any, Awaitable, Callable, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import numpy as np
import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
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
class MockSessionAVSettings:
    audio_input: MockSessionAudioSettings = MockSessionAudioSettings()
    audio_output: MockSessionAudioSettings = MockSessionAudioSettings()


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

# Mock the module in sys.modules so imports work
sys.modules["vonage_video_connector"] = vonage_video_mock
sys.modules["vonage_video_connector.models"] = vonage_video_mock.models


# Now we can import the transport classes since the vonage_video module is mocked
from pipecat.transports.vonage.video_webrtc import (
    AudioProps,
    VonageClient,
    VonageClientListener,
    VonageClientParams,
    VonagePublisherSettings,
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
        self.VonageClientParams = VonageClientParams
        self.VonagePublisherSettings = VonagePublisherSettings
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

    def test_vonage_client_params_defaults(self) -> None:
        """Test VonageClientParams default values."""
        params = self.VonageClientParams()
        assert params.audio_in_sample_rate == 48000
        assert params.audio_in_channels == 2
        assert params.enable_migration is False

    def test_vonage_client_params_custom_values(self) -> None:
        """Test VonageClientParams with custom values."""
        params = self.VonageClientParams(
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_out_sample_rate=22050,
            audio_out_channels=1,
            enable_migration=True,
        )
        assert params.audio_in_sample_rate == 16000
        assert params.audio_in_channels == 1
        assert params.audio_out_sample_rate == 22050
        assert params.audio_out_channels == 1
        assert params.enable_migration is True

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

        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        assert client._application_id == self.application_id
        assert client._session_id == self.session_id
        assert client._token == self.token
        assert client._params == params
        assert client._connected is False
        assert client._connection_counter == 0
        vonage_video_mock.VonageVideoClient.assert_called_once()

    def test_vonage_client_add_remove_listener(self) -> None:
        """Test adding and removing listeners from VonageClient."""
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        listener = self.VonageClientListener()
        listener_id = client.add_listener(listener)

        assert isinstance(listener_id, int)
        assert listener_id in client._listeners
        assert client._listeners[listener_id] == listener

        client.remove_listener(listener_id)
        assert listener_id not in client._listeners

    def _setup_audio_ready_callback(self, client: VonageClient) -> None:
        """Helper to set up the audio ready callback."""

        def connect_side_effect(*_: Any, **__: Any) -> bool:
            client._on_session_ready_for_audio_cb(vonage_video_mock.models.Session())
            return True

        self.mock_client_instance.connect = MagicMock(side_effect=connect_side_effect)

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
        params = self.VonageClientParams()

        # make changes to params depending on the configuration to check the right value
        # goes to the right destination
        params.audio_in_channels = 1 if has_video else 2
        params.audio_out_channels = 2 if has_video else 1
        params.audio_in_sample_rate = 44100 if has_video else 22050
        params.audio_out_sample_rate = 22050 if has_video else 44100
        params.enable_migration = has_video

        publisher_settings = self.VonagePublisherSettings(has_audio=has_audio, has_video=has_video)
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

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
            ),
            enable_migration=params.enable_migration,
            logging=MockLoggingSettings(level="INFO"),
        )
        assert call_args[1]["on_audio_data_cb"] == client._on_session_audio_data_cb
        assert call_args[1]["on_error_cb"] == client._on_session_error_cb
        assert call_args[1]["on_connected_cb"] == client._on_session_connected_cb
        assert call_args[1]["on_disconnected_cb"] == client._on_session_disconnected_cb
        assert call_args[1]["on_ready_for_audio_cb"] == client._on_session_ready_for_audio_cb
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
        params = self.VonageClientParams()

        # make changes to params depending on the configuration to check the right value
        # goes to the right destination
        params.audio_in_channels = 1 if has_video else 2
        params.audio_out_channels = 2 if has_video else 1
        params.audio_in_sample_rate = 44100 if has_video else 22050
        params.audio_out_sample_rate = 22050 if has_video else 44100
        params.enable_migration = has_video

        publisher_settings = self.VonagePublisherSettings(
            has_audio=has_audio,
            has_video=has_video,
            enable_stereo_mode=has_video,
            enable_opus_dtx=not has_video,
        )
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True

        # only set this callback if we have audio enabled
        if has_audio:
            self._setup_audio_ready_callback(client)
        await client.connect()

        self.mock_client_instance.connect.assert_called_once()

        # trigger the _on_session_connected_cb to simulate being connected
        client._on_session_connected_cb(vonage_video_mock.models.Session())

        # if no audio and no video, publish should not be called
        if not has_audio and not has_video:
            self.mock_client_instance.publish.assert_not_called()
            return

        # Verify publish was called with correct parameters
        self.mock_client_instance.publish.assert_called_once()
        call_args = self.mock_client_instance.publish.call_args
        assert call_args[1]["settings"] == MockPublisherSettings(
            name=publisher_settings.name,
            audio_settings=MockPublisherAudioSettings(
                enable_stereo_mode=publisher_settings.enable_stereo_mode,
                enable_opus_dtx=publisher_settings.enable_opus_dtx,
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
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings(has_audio=True)
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        self._setup_audio_ready_callback(client)

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True

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
    async def test_vonage_client_concurrent_connects(self) -> None:
        """Test VonageClient concurrent connects."""
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings(has_audio=True)
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True

        # create a listener
        listener = self.VonageClientListener()
        listener.on_connected = AsyncMock()
        client.add_listener(listener)

        # send two parallel connect calls and let them get stuck awaiting
        connect1_task = asyncio.create_task(client.connect())
        connect2_task = asyncio.create_task(client.connect())

        # wait for the publish_ready promise be created
        # Wait for both tasks to reach the await (they will be pending on _publish_ready)
        while not client._publish_ready:
            await asyncio.sleep(0.01)

        # Now both connects are waiting on the same promise, we can set it to complete
        client._on_session_ready_for_audio_cb(vonage_video_mock.models.Session())

        # await for the connections to now complete
        await asyncio.gather(connect1_task, connect2_task)

        # SDK connect should only be called once
        self.mock_client_instance.connect.assert_called_once()
        listener.on_connected.assert_called_once()

    @pytest.mark.asyncio
    async def test_vonage_client_connect_failure(self) -> None:
        """Test VonageClient connect method when connection fails."""
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        # Mock the connect method to return False
        self.mock_client_instance.connect.return_value = False

        with pytest.raises(Exception) as exc_info:
            await client.connect()

        assert "Could not connect to session" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_vonage_client_disconnect_before_connecting(self) -> None:
        """Test VonageClient disconnect method before connecting."""
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        listener = self.VonageClientListener()
        listener.on_disconnected = AsyncMock()
        client.add_listener(listener)

        await client.disconnect()

        self.mock_client_instance.disconnect.assert_not_called()
        listener.on_disconnected.assert_not_called()

    @pytest.mark.asyncio
    async def test_vonage_client_disconnect(self) -> None:
        """Test VonageClient disconnect method."""
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings(has_audio=True)
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True
        self._setup_audio_ready_callback(client)

        # create a listener
        listener = self.VonageClientListener()
        listener.on_disconnected = AsyncMock()
        client.add_listener(listener)

        # send two parallel connect calls and let them get stuck awaiting
        connect_promise1 = client.connect()
        connect_promise2 = client.connect()

        # await for the connections to now complete
        await asyncio.gather(connect_promise1, connect_promise2)

        # send the first disconnect call, we should still be connected
        await client.disconnect()
        self.mock_client_instance.disconnect.assert_not_called()
        listener.on_disconnected.assert_not_called()

        # check the second disconnect now disconnects for real
        await client.disconnect()
        self.mock_client_instance.disconnect.assert_called_once()
        listener.on_disconnected.assert_called_once()

        # an extra disconnect should not do anything
        await client.disconnect()
        self.mock_client_instance.disconnect.assert_called_once()
        listener.on_disconnected.assert_called_once()

    @pytest.mark.asyncio
    async def test_vonage_client_write_audio(self) -> None:
        """Test VonageClient write_audio method."""
        params = self.VonageClientParams(audio_out_channels=2, audio_out_sample_rate=48000)
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        # Create mock audio data
        audio_data = b"\x00\x01\x02\x03\x04\x05\x06\x07"  # 4 frames of 2-channel 16-bit audio

        await client.write_audio(audio_data)

        self.mock_client_instance.add_audio.assert_called_once()
        call_args = self.mock_client_instance.add_audio.call_args[0][0]
        assert call_args.number_of_frames == 2  # 8 bytes / (2 channels * 2 bytes)
        assert call_args.number_of_channels == 2
        assert call_args.sample_rate == 48000

    @pytest.mark.asyncio
    async def test_vonage_client_events(self) -> None:
        """Test VonageClient events"""
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings(has_audio=True)
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        # Mock the connect method to return True
        self.mock_client_instance.connect.return_value = True
        self._setup_audio_ready_callback(client)

        # create a listener
        listener = self.VonageClientListener()
        on_error_mock = AsyncMock()
        listener.on_error = on_error_mock
        on_audio_in_mock = MagicMock()
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

        listener.on_audio_in.assert_called_once_with(session, mock_audio_data)
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
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_input_transport_initialization(self, mock_resampler: MagicMock) -> None:
        """Test VonageVideoWebrtcInputTransport initialization."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        transport_params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)

        assert transport._client == client
        assert transport._initialized is False
        mock_resampler.assert_called_once()

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_input_transport_start(self, mock_resampler: MagicMock) -> None:
        """Test VonageVideoWebrtcInputTransport start method."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        transport_params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)

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
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_input_transport_stop(self, mock_resampler: MagicMock) -> None:
        """Test VonageVideoWebrtcInputTransport stop method."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        transport_params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)
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
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_input_transport_cancel(self, mock_resampler: MagicMock) -> None:
        """Test VonageVideoWebrtcInputTransport cancel method."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        transport_params = self.VonageVideoWebrtcTransportParams(audio_in_enabled=True)
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)
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

    async def create_output_transport(
        self, params: VonageVideoWebrtcTransportParams
    ) -> VonageVideoWebrtcOutputTransport:
        publisher_settings = VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id,
            self.session_id,
            self.token,
            self.VonageClientParams(),
            publisher_settings,
        )
        transport = VonageVideoWebrtcOutputTransport(client, params)

        clock: SystemClock = SystemClock()  # type: ignore[no-untyped-call]
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_event_loop()))
        transport_params = self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        transport = self.VonageVideoWebrtcOutputTransport(client, transport_params)
        await transport.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))

        return transport

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_output_transport_initialization(self, mock_resampler: MagicMock) -> None:
        """Test VonageVideoWebrtcOutputTransport initialization."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        transport_params = self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        transport = self.VonageVideoWebrtcOutputTransport(client, transport_params)

        assert transport._client == client
        assert transport._initialized is False
        mock_resampler.assert_called_once()

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_output_transport_start(self, mock_resampler: MagicMock) -> None:
        """Test VonageVideoWebrtcOutputTransport start method."""
        mock_resampler.return_value = Mock()
        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

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
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_output_transport_write_audio_frame(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test VonageVideoWebrtcOutputTransport write_audio_frame method."""
        mock_resampler_instance = Mock()
        mock_resampler_instance.resample = AsyncMock(return_value=b"\x00\x01\x02\x03")
        mock_resampler.return_value = mock_resampler_instance

        params = self.VonageClientParams(audio_out_sample_rate=48000, audio_out_channels=2)
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        with (
            patch.object(client, "write_audio", AsyncMock()) as client_write_audio_mock,
            patch.object(client, "get_params", Mock(return_value=params)),
        ):
            transport_params = self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
            transport = self.VonageVideoWebrtcOutputTransport(client, transport_params)
            transport._connected = True

            # Create a mock audio frame
            audio_frame = OutputAudioRawFrame(
                audio=b"\x00\x01\x02\x03", sample_rate=16000, num_channels=1
            )

            await transport.write_audio_frame(audio_frame)

            # Verify resampling was called
            mock_resampler_instance.resample.assert_called_once_with(
                audio_frame.audio, 16000, 48000
            )
            # Verify audio was written to client
            client_write_audio_mock.assert_called_once()

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_output_transport_process_frame_with_interruption(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test VonageVideoWebrtcOutputTransport process_frame method with InterruptionFrame."""
        mock_resampler.return_value = Mock()
        transport = await self.create_output_transport(
            params=self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        )
        client = transport._client

        with (
            patch.object(client, "clear_media_buffers") as clear_buffers_mock,
            patch.object(client, "connect", AsyncMock()),
        ):
            interruption_frame = InterruptionFrame()
            await transport.start(StartFrame())
            await transport.process_frame(interruption_frame, FrameDirection.DOWNSTREAM)

            # Verify clear_media_buffers was called
            clear_buffers_mock.assert_called_once()

    @pytest.mark.asyncio
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_output_transport_process_frame_without_interruption(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test VonageVideoWebrtcOutputTransport process_frame method with non-interruption frame."""
        mock_resampler.return_value = Mock()
        transport = await self.create_output_transport(
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
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_output_transport_process_frame_when_not_connected(
        self, mock_resampler: MagicMock
    ) -> None:
        """Test VonageVideoWebrtcOutputTransport process_frame method when not connected."""
        mock_resampler.return_value = Mock()
        transport = await self.create_output_transport(
            params=self.VonageVideoWebrtcTransportParams(audio_out_enabled=True)
        )
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

        transport = self.VonageVideoWebrtcTransport(
            self.application_id, self.session_id, self.token, params
        )

        assert transport._client is not None
        assert transport._one_stream_received is False

        # Verify vonage client was initialized with correct parameters
        client_params = transport._client._params
        assert client_params.audio_out_sample_rate == 48000
        assert client_params.audio_out_channels == 2
        assert client_params.enable_migration is True

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
    @patch("pipecat.transports.vonage.video_webrtc.asyncio.run_coroutine_threadsafe")
    @patch("pipecat.transports.vonage.video_webrtc.create_stream_resampler")
    async def test_vonage_input_audio_callback(
        self, mock_resampler: MagicMock, mock_run_coroutine: MagicMock
    ) -> None:
        """Test audio input callback processing."""
        resampled_audio = b"\x00\x01\x02\x03"
        resampled_bitrate = 26000
        mock_resampler_instance = Mock()
        mock_resampler_instance.resample = AsyncMock(return_value=resampled_audio)
        mock_resampler.return_value = mock_resampler_instance

        push_frame_coroutine: Optional[Awaitable[None]] = None

        # Mock the run_coroutine_threadsafe to capture the coroutine
        def mock_run_coro(coro: Awaitable[None], loop: asyncio.AbstractEventLoop) -> Future[None]:
            nonlocal push_frame_coroutine
            push_frame_coroutine = coro
            # Return a mock task
            task = Mock(spec=Future[None])
            task.result.return_value = None
            return task

        mock_run_coroutine.side_effect = mock_run_coro

        params = self.VonageClientParams()
        publisher_settings = self.VonagePublisherSettings()
        client = self.VonageClient(
            self.application_id, self.session_id, self.token, params, publisher_settings
        )

        transport_params = self.VonageVideoWebrtcTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=resampled_bitrate,
        )
        transport = self.VonageVideoWebrtcInputTransport(client, transport_params)
        transport._listener_id = 1
        with (
            patch.object(transport, "push_audio_frame", AsyncMock()) as mock_push_audio_frame,
            patch.object(transport, "get_event_loop", Mock(return_value=asyncio.get_event_loop())),
            patch.object(client, "connect", AsyncMock(return_value=1)),
            patch.object(transport, "set_transport_ready", AsyncMock()),
        ):
            start_frame = StartFrame()
            await transport.start(start_frame)

            # Create mock audio data
            audio_buffer = np.array([100, 200, 300, 400], dtype=np.int16)
            mock_audio_data = Mock()
            mock_audio_data.sample_buffer = audio_buffer.tobytes()
            mock_audio_data.number_of_frames = 2
            mock_audio_data.number_of_channels = 2
            mock_audio_data.sample_rate = 48000

            # Create mock session
            mock_session = Mock()

            # Call the audio callback
            transport._audio_in_cb(mock_session, mock_audio_data)

            # Execute the captured coroutine and check it does what we expect
            assert push_frame_coroutine is not None
            await push_frame_coroutine

            mock_push_audio_frame.assert_called_once()
            # Verify run_coroutine_threadsafe was called
            mock_run_coroutine.assert_called_once()
            arg = mock_push_audio_frame.call_args[0][0]
            assert isinstance(arg, InputAudioRawFrame)
            assert arg.audio == resampled_audio
            assert arg.sample_rate == resampled_bitrate
            assert arg.num_channels == 1

    @pytest.mark.asyncio
    async def test_vonage_transport_event_handlers(self) -> None:
        """Test VonageVideoWebrtcTransport event handlers."""
        params = self.VonageVideoWebrtcTransportParams()
        transport = self.VonageVideoWebrtcTransport(
            self.application_id, self.session_id, self.token, params
        )

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
        transport = self.VonageVideoWebrtcTransport(
            self.application_id, self.session_id, self.token, params
        )

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
