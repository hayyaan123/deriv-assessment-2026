# API Rate Limits

## Limits by Plan

- Free plan: 10 requests/second, burst up to 20.
- Growth plan: 50 requests/second, burst up to 100.
- Enterprise plan: custom limits negotiated per contract.

## Exceeding the Limit

If you exceed your rate limit, the API returns an HTTP `429 Too Many Requests` response. The
response includes a `Retry-After` header indicating how many seconds to wait before retrying.
Requests that are rate-limited are not queued or processed later — they are rejected outright
and must be resent by the client.

## Monitoring Usage

Current rate limit status is included in every API response via the `X-RateLimit-Remaining`
and `X-RateLimit-Reset` headers, so clients can proactively throttle themselves before
hitting a 429.

## Repeated Violations

Accounts that are rate-limited more than 100 times in a rolling 24-hour window may receive
a warning email. There is no automatic account suspension for rate limit violations alone.
