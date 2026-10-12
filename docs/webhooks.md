# Webhooks

## Overview

Webhooks let your application receive real-time notifications when events occur, such as
payment completion, subscription changes, or data export completion.

## Delivery

Webhook events are delivered via HTTP POST to the endpoint URL configured in your dashboard.
Each delivery includes a signature header (`X-Webhook-Signature`) so you can verify the
payload originated from our platform.

## Retry Policy

If your endpoint does not respond with a `2xx` status code, the event is retried up to 5
times with exponential backoff, over a maximum window of 24 hours.

## Delivery Log Retention

Webhook delivery logs, including the request payload, response status, and response body,
are retained for **30 days**. After 30 days, logs are permanently deleted and cannot be
recovered. You can export logs to CSV before they expire.

## Supported Event Types

The platform currently supports webhook events for: payments, subscriptions, user account
changes, and data export jobs. Real-time push protocols other than webhooks (such as
WebSockets) are not covered by this document.
