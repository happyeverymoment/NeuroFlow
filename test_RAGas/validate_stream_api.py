import argparse
import json
from typing import Dict, List

import requests


def validate_events(events: List[Dict]) -> List[str]:
    errors: List[str] = []
    if not events:
        return ["响应为空，未收到任何 NDJSON 事件。"]

    event_types = [str(e.get("type", "")) for e in events]
    if "meta" not in event_types:
        errors.append("缺少 type=meta 事件。")
    if "done" not in event_types:
        errors.append("缺少 type=done 事件。")
    if "delta" not in event_types:
        errors.append("缺少 type=delta 事件。")

    meta_events = [e for e in events if e.get("type") == "meta"]
    done_events = [e for e in events if e.get("type") == "done"]
    delta_events = [e for e in events if e.get("type") == "delta"]

    for meta in meta_events:
        if "route" not in meta:
            errors.append("meta 事件缺少 route 字段。")
        intent = meta.get("intent")
        if not isinstance(intent, dict):
            errors.append("meta 事件缺少 intent 对象。")
        elif "module" not in intent:
            errors.append("meta.intent 缺少 module 字段。")

    for delta in delta_events:
        if "text" not in delta:
            errors.append("delta 事件缺少 text 字段。")

    for done in done_events:
        if "answer" not in done:
            errors.append("done 事件缺少 answer 字段。")
        if "route" not in done:
            errors.append("done 事件缺少 route 字段。")

    if done_events and delta_events:
        delta_text = "".join(str(d.get("text", "")) for d in delta_events).strip()
        done_answer = str(done_events[-1].get("answer", "")).strip()
        if delta_text and done_answer and delta_text != done_answer:
            errors.append("delta 拼接文本与 done.answer 不一致。")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NDJSON stream API schema.")
    parser.add_argument("--url", required=True, help="Streaming endpoint URL")
    parser.add_argument("--message", default="你好", help="Input message")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout seconds")
    args = parser.parse_args()

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/x-ndjson",
    }
    payload = {"message": args.message, "stream": True}

    print(f"[INFO] POST {args.url}")
    print(f"[INFO] payload={payload}")
    response = requests.post(args.url, json=payload, headers=headers, timeout=args.timeout, stream=True)
    response.raise_for_status()

    events: List[Dict] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        event = json.loads(line)
        events.append(event)
        print(json.dumps(event, ensure_ascii=False))

    errors = validate_events(events)
    if errors:
        print("[FAIL] 接口校验失败：")
        for err in errors:
            print(f" - {err}")
        raise SystemExit(1)

    print("[OK] 接口校验通过。")
    done_event = [e for e in events if e.get("type") == "done"][-1]
    print(f"[OK] route={done_event.get('route')}")
    print(f"[OK] answer={done_event.get('answer')}")


if __name__ == "__main__":
    main()
