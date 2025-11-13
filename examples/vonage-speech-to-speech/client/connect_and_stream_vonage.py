#!/usr/bin/env python3
"""
Use a Vonage (OpenTok) Video API existing session, generate a token,
and connect its audio to your Pipecat WebSocket endpoint.
"""

import argparse
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from vonage import Auth, HttpClientOptions, Vonage
from vonage_video import AudioConnectorOptions, TokenOptions

# ---- helpers ----------------------------------------------------------------


def parse_kv_pairs(items: List[str]) -> Dict[str, str]:
    """
    Parse CLI --header/--param entries like "Key=Value" or "Key:Value".
    """
    out: Dict[str, str] = {}
    for raw in items or []:
        sep = "=" if "=" in raw else (":" if ":" in raw else None)
        if not sep:
            raise ValueError(f"Invalid header/param format: {raw!r}. Use Key=Value")
        k, v = raw.split(sep, 1)
        out[k.strip()] = v.strip()
    return out


def comma_list(s: str | None) -> List[str]:
    return [x.strip() for x in s.split(",")] if s else []


# ---- main -------------------------------------------------------------------


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(
        description="Create a session and connect its audio to a WebSocket (Pipecat)."
    )
    # Auth
    p.add_argument("--application-id", default=os.getenv("VONAGE_APPLICATION_ID"), required=False)
    p.add_argument("--private-key", default=os.getenv("VONAGE_PRIVATE_KEY"), required=False)

    # Where to connect
    p.add_argument("--ws-uri", default=os.getenv("WS_URI"), help="wss://...", required=False)
    p.add_argument("--audio-rate", type=int, default=int(os.getenv("VONAGE_AUDIO_RATE", "16000")))
    bidirectional_env = os.getenv("VONAGE_BIDIRECTIONAL")
    if bidirectional_env is not None:
        if bidirectional_env.lower() not in ("true", "false"):
            raise SystemExit("VONAGE_BIDIRECTIONAL must be 'true' or 'false'")
        bidirectional_default = bidirectional_env.lower() == "true"
    else:
        bidirectional_default = True

    p.add_argument("--bidirectional", action="store_true", default=bidirectional_default)

    # An existing session which needs to be connected to pipecat-ai
    p.add_argument("--session-id", default=os.getenv("VONAGE_SESSION_ID"))

    # Optional streams and headers (to pass to the WS)
    p.add_argument(
        "--streams", default=os.getenv("VONAGE_STREAMS"), help="Comma-separated stream IDs"
    )
    p.add_argument(
        "--header",
        action="append",
        help="Extra header(s) for WS, e.g. --header X-Foo=bar (repeatable)",
    )

    # Optional: choose API base. If your SDK doesn’t accept api_url, set VONAGE_API_URL env before run.
    p.add_argument("--api-base", default=os.getenv("VONAGE_API_URL", "api.vonage.com"))

    args = p.parse_args()

    # Validate inputs
    missing = [
        k
        for k, v in {
            "application-id": args.application_id,
            "private-key": args.private_key,
            "ws-uri": args.ws_uri,
            "session-id": args.session_id,
        }.items()
        if not v
    ]
    if missing:
        raise SystemExit(f"Missing required args/env: {', '.join(missing)}")

    # Create an Auth instance
    auth = Auth(
        application_id=args.application_id,
        private_key=args.private_key,
    )

    # Create HttpClientOptions instance
    # (not required unless you want to change options from the defaults)
    options = HttpClientOptions(video_host="video." + args.api_base, timeout=30)

    # Create a Vonage instance
    vonage = Vonage(auth=auth, http_client_options=options)

    session_id = args.session_id
    print(f"Using existing session: {session_id}")

    # Token: generate a fresh one tied to this session
    token_options = TokenOptions(session_id=session_id, role="publisher")
    token = vonage.video.generate_client_token(token_options)
    print(f"Generated token: {token[:32]}...")  # don’t print full token in logs

    # Build websocket options (mirrors your Postman body)
    ws_opts = {
        "uri": args.ws_uri,
        "audioRate": args.audio_rate,
        "bidirectional": bool(args.bidirectional),
    }

    # Optional stream filtering
    stream_list = comma_list(args.streams)
    if stream_list:
        ws_opts["streams"] = stream_list

    # Optional headers passed to your WS server
    headers = parse_kv_pairs(args.header or [])
    if headers:
        ws_opts["headers"] = headers

    print("Connecting audio to WebSocket with options:")
    print(json.dumps(ws_opts, indent=2))

    # Call the Audio Connector (this is equivalent to POST /v2/project/{apiKey}/connect)
    audio_connector_options = AudioConnectorOptions(
        session_id=session_id, token=token, websocket=ws_opts
    )
    resp = vonage.video.start_audio_connector(audio_connector_options)

    # The SDK returns a small object/dict; print it for visibility
    try:
        print("Connect response:", json.dumps(resp, indent=2))
    except TypeError:
        # Not JSON-serializable; just repr it
        print("Connect response:", resp)

    print("\nSuccess! Your Video session should now stream audio to/from:", args.ws_uri)


if __name__ == "__main__":
    main()
