# Security policy

## Reporting a problem

Please don't open a public issue for a security problem. Report it privately instead:

- with GitHub's **Report a vulnerability** button on this repo's Security tab, or
- by email to hello@zhagaramlabs.com, with "Security: oosai" in the subject.

Say what you found, how to reproduce it, and what it could let someone do. We'll reply as soon as we
can, keep you updated until it's fixed, and credit you if you'd like.

## What counts

- The voice agent's code, and the local servers it starts (they should listen on 127.0.0.1 only).
- A way to make the agent run commands, read files or reveal private data.
- A wrong or unsafe answer on its own isn't a security problem: open a "Wrong or unsafe answer"
  issue instead.
- Secrets, personal data or recordings that shouldn't be in this repo or its releases.

## Versions

We fix the latest code on `main`.
