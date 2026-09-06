---
name: voice
applies_to: modality=voice
description: What a scenario has to account for when the person reaches the agent by speaking. Read alongside the scenario-writing instructions whenever the contract says the modality is voice.
---

# Writing scenarios for a voice agent

A voice agent is reached by a person speaking, in real time, who cannot see anything. That person
answers several questions in one breath, corrects themselves mid-sentence, mishears a digit, talks
over a confirmation, and sometimes goes silent. Every requirement below follows from one of those
facts, and none of them replaces the general requirements a scenario has to meet.

Whether the agent placed this call or answered it changes how the person is written. The contract
carries that as `CALL DIRECTION`, and the general instructions say what each direction requires: read
it there rather than deciding it here.

## What a voice scenario can test that a chat one cannot

- **The caller answers three questions in one breath**, in their own order, before being asked.
  Real callers do this constantly. An agent that collects one field per turn fails here and passes
  every written test.
- **The caller changes their mind mid-sentence**, and the correction lands after the original. The
  second value is the real one.
- **A value has to be read back and heard.** Codes, prices, times. A digit misheard is a real
  failure, and it only exists out loud.
- **Interruption.** The caller talks over the agent's confirmation. What the agent believes was
  confirmed is now a question.
- **Silence.** The caller goes quiet, or the line is noisy and they ask for something again.

A suite of voice scenarios that could all have been typed has not tested the modality.

## An attempted transfer is not a completed one

A voice run may record that the agent tried to hand the call to a person without the receiving side
ever picking it up, because completing that handoff needs telephony the run does not have. So a
scenario about handing off tests the offer or the attempt, and the work that would follow it belongs
in a separate scenario. A sub-goal that only holds once somebody answers will fail on a correct
handoff.

## What this modality lets you vary

`background_noise` is per scenario, not a suite setting. Choose it from the situation rather than
sprinkling it: a caller in a vehicle, a caller in an office, a caller in a crowd. A quiet scenario is
the control that makes a noisy one mean something, so a suite needs both.

Accent and language belong to who the caller is, and they change what the agent's transcription has
to survive. They are dealt across the suite; take the one you are given unless the scenario genuinely
needs another.

`max_turns` is a budget, not a target. A scenario that needs eighteen turns to reach the thing it
tests is fine. One that spends eighteen turns being polite is not.

## What does not belong in a voice instruction

Never write stage directions. No *sighs*, no [annoyed]. Anything in brackets is read aloud, so the
caller says the word "annoyed" instead of sounding it. Manner comes from the persona's disposition.

Never tell the caller what the agent should do. They are on the phone, not reading the contract.
