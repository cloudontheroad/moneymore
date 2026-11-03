"""Read-only localhost acceptance; requires an actual browser screenshot record."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from moneymore.research.s3_review import read_public_review


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (args.output / 'results.json').exists():
        raise FileExistsError('IMMUTABLE_SERVING_ACCEPTANCE')
    browser = json.loads((args.output / 'browser.json').read_text(encoding='utf-8'))
    if not all(browser.get(k) for k in ['review_visible', 'blocked_title_visible', 'six_comparison_rows_visible']):
        raise ValueError('ACTUAL_BROWSER_ACCEPTANCE_REQUIRED')
    if hashlib.sha256((args.output / 'review.png').read_bytes()).hexdigest() != browser['screenshot_sha256']:
        raise ValueError('SCREENSHOT_CHANGED')
    with urlopen('http://127.0.0.1:8788/api/health', timeout=10) as response:
        health = json.load(response)
    with urlopen('http://127.0.0.1:8788/api/qlib-challenger', timeout=45) as response:
        public = json.load(response)['style_research_review']
    root = Path(__file__).resolve().parents[1]
    packet = root / 'state/style-balance/s3-release-review.json'
    if public != read_public_review(packet):
        raise ValueError('RUNNING_API_NOT_USING_CURRENT_EVIDENCE')
    if health['mode'] != 'PAPER_ONLY' or not health['scheduler'] or public['observation_authorized'] or public['real_orders_enabled']:
        raise ValueError('SERVICE_SAFETY_OR_SCHEDULER_CHECK_FAILED')
    if health['realtime_data_source'] != 'alphafeed' or health['research_data_source'] != 'tushare':
        raise ValueError('UNEXPECTED_DATA_PROVIDER')
    with urlopen('http://127.0.0.1:3000/', timeout=15) as response:
        if response.status != 200 or 'MoneyMore' not in response.read().decode('utf-8'):
            raise ValueError('DASHBOARD_NOT_SERVED')
    result = {
        'status': 'LOCAL_API_AND_BROWSER_VERIFIED', 'observed_at': datetime.now(timezone.utc).isoformat(),
        'public_payload_sha256': hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest(),
        'review_packet_sha256': hashlib.sha256(packet.read_bytes()).hexdigest(),
        'browser_evidence_sha256': hashlib.sha256((args.output / 'browser.json').read_bytes()).hexdigest(),
        'health': {k: health[k] for k in ['status', 'mode', 'scheduler', 'weekly_training_scheduler', 'realtime_data_source', 'research_data_source']},
        'api_review_status': public['status'], 'browser_screenshot_verified': True,
        'real_orders_enabled': False, 'new_candidate_created': False,
    }
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
