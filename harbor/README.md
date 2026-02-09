# Harbor Training Guide
Example for using [Harbor](https://harborframework.com), an eval framework for agents, with Modal Sandboxes.
Quickly scale up thousands of instances in a warm pool without hitting Modal rate limits.

## How do rate limits work?
Modal limits Sandbox resource requests, or tokens, using the [token bucket algorithm](https://en.wikipedia.org/wiki/Token_bucket). Each token draws from a fixed-size token bucket until the bucket's balance reaches 0, and a [Resource Exhausted Error](https://modal.com/docs/reference/modal.exception#modalexceptionresourceexhaustederror) error is thrown.

Tokens are refilled at a constant rate of **5 tokens/sec**. When many sandboxes are created at once, which we need for Harbor agent rollouts, a burst credit of **150** tokens is given to the client. 

 