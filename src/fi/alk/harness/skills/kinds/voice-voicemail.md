---
name: voice-voicemail
applies_to: modality=voice,voicemail=on
description: What a scenario has to account for when a mailbox answers an outbound call instead of a person. Read alongside the voice instructions, and only on a run where mailbox scenarios are allowed.
---

## When a mailbox answers instead of a person

An outbound call reaches voicemail often, and an agent that runs its interactive script at a
recording is a real defect nobody hears about until a customer does. Set `answered_by: voicemail` on
an outbound scenario and the person is replaced by a mailbox: it plays its greeting once and then
says nothing at all, whatever the agent asks.

What is being tested is entirely on the agent's side. Does it notice it is talking to a machine
rather than waiting for answers that will never come. Does it leave a message that stands on its own,
with who is calling, why, and what happens next, rather than the first line of a conversation. Does
it stop, instead of holding the line open asking questions.

Which kind of mailbox is `voicemail_style`, and the four are four different tests:

- `personal`, the person's own greeting followed by a tone. The ordinary case, and the only one that
  carries their name, so it is also the one where the agent can confirm who it reached.
- `carrier`, the network default, which names nobody. The agent has no confirmation of who answered
  and has to leave a message anyway.
- `operator`, a long formal announcement before the tone. This is the one an agent that starts
  talking too early speaks over, so its message is recorded half missing.
- `full`, a mailbox that cannot record. **There is no tone at all**, and the right behaviour is to
  recognise there is nowhere to leave a message and end the call to try later, not to talk into
  nothing.

**State the style. Do not leave it out.** Left out it falls back to `personal`, and personal is the
easiest of the four: the greeting names the person, so the agent can confirm who it reached and the
tone tells it when to talk. Most suites are only large enough for one mailbox, and if that one always
defaults, the three harder cases never get tested at all: `carrier` where nothing confirms who
answered, `operator` where the announcement is long enough to be talked over, and `full` where there
is no tone and the right move is to give up rather than leave a message into nothing.

So choose the style from the situation, the way you choose everything else. A number nobody has ever
confirmed belongs to this person is a carrier mailbox. A work line reached out of hours is an operator
system. Somebody who has been letting calls go for a week has a full one. Where you write two or more
mailboxes, use at least two styles. A greeting in another language is worth one of them too, since its
transcription has to survive.

Two things follow from a mailbox not being a person. The scenario's persona still carries the
greeting as its opening line, so write the greeting there. And the caller cannot supply anything, so
a mailbox scenario never asks the agent to collect a value, confirm a detail or reach agreement:
those belong in a scenario where somebody picks up.

That applies to the sub-goals as hard as it does to the situation, and it is where these scenarios go
wrong in practice. Two measured mailbox calls failed on `check_booking_status` alone, because the agent
only reaches that tool after the person it called has spoken, and on a mailbox nobody ever does. Every
sub-goal on a mailbox scenario has to be something the agent can do with nobody on the line: it
recognised a machine, the message it left says who is calling and why, it stopped instead of asking
questions. A sub-goal that needs an answer marks a correctly handled mailbox as a failure and tells
you nothing.

Keep these rare: at most one scenario in twenty, and none at all is a perfectly good suite. They test
one narrow thing well, and a suite full of mailboxes has stopped testing the agent talking to people.

**Nothing else goes on a mailbox scenario.** No `background_noise`, and no `bystander`. What the agent
reaches is a recording played back by a switch, so there is no room to overhear and nobody in it to
interrupt, and either one would tell the agent it is talking to a person when the whole point is that
it is not.

