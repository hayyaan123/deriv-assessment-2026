# Authentication

## Supported Methods

The platform supports two authentication methods for API access:

1. **API Keys** — Static keys generated from the developer dashboard. Suitable for
   server-to-server integrations. Keys should be passed in the `Authorization: Bearer <key>`
   header on every request.
2. **OAuth 2.0** — Recommended for applications acting on behalf of a user. Supports the
   authorization code grant and client credentials grant. Access tokens expire after 1 hour;
   refresh tokens are valid for 30 days.

## Key Rotation

API keys can be rotated at any time from the dashboard. Old keys remain valid for 24 hours
after a new key is issued, to allow for a graceful transition.

## Security Notes

All API requests must be made over HTTPS. Requests made over plain HTTP are rejected with a
`426 Upgrade Required` status code.
