---
name: callers
description: Who the simulated caller is and how it behaves - persona fields and offered values, languages and what can be heard, accents and voices, background noise, temperament, mood and frustration, interruptions, holds, and what a caller cannot produce.
---

# The simulated caller

The caller is a model given the scenario's situation and persona plus fixed rules for how a real
person behaves. On a voice call its words are spoken by text-to-speech and the agent's words reach it
through speech-to-text. Always call `offered_values` for the live lists; this topic explains them.

## Persona

One believable person, not a set of fields: name, language, home and accent are chosen together.

| Field | Notes |
|---|---|
| name | Common, real given and family name, unique in the suite; never famous or fictional |
| gender, age group | Offered values; on voice they also shape the voice |
| location | Where they call from (offered countries) |
| personality, communication style | Offered values; drive temperament and speech |
| occupation | Offered value |
| languages | Exactly one: the language spoken on the call |
| accent | How English is spoken; also picks the voice. Forced to Neutral when the language is not English |
| initial message | What the first request is about; the caller still says it in its own words |

On a chat agent, accent and background noise do not exist and are cleared.

## Languages: what can be heard

Three pieces: the caller's brain (a model, any language), its voice (one language per call), and its
ears (speech-to-text on the agent: one fixed language, or a multilingual mode that hears English mixed
with one of nine common languages). A caller in a language outside that mix can only hear an agent
answering in the same language; if the agent answers in English, the caller cannot hear it and the call
fails for a reason that is not the agent's. A persona speaks one language; a "bilingual caller" speaks
its first language only.

## Voice

The accent picks a voice bucket first (an accent without its own voices gets a general English voice and
keeps the accent only in its words), then the language, then English. Gender picks within the bucket
and the persona's name picks the voice, so a caller keeps its voice across runs. Speech rate is derived
from the persona.

## Background noise

Voice only. A recording of a place (street, vehicle, transit, airport, shop or mall, office, home,
hospital, outdoors, crowd) is mixed under the caller's voice, levelled so it sits well under speech,
louder on scenarios that are about the noise. Within a run, calls from one place rotate through its
recordings. The caller is told where it is calling from and answers as someone there. A quiet line
is the control and is kept rare. The bed never speaks: a second voice or an announcement cannot be
produced.

## Behaviour

- Uses only the facts given; never invents an identifier the agent will look up.
- Answers what was asked, one fact at a time; lets the agent finish; follows sequence words literally.
- Plain and matter-of-fact, not an assistant: no praising answers, thanks at most once at the end.
- Holds silently while the agent checks; says once that it is still there after a long pause.
- If the agent repeats itself three times without progress, says so once and ends the call.
- Closes in one turn once the outcome is confirmed; nothing is spoken after its goodbye.
- The scenario outranks the rules: a hostile or confused person is played that way, but it never
  invents a looked-up identifier, never claims to have done something away from the call, and the call
  always ends.

## Temperament, mood and frustration

The persona's personality sets a temperament (calm, driven and anxious components) that drives
patience and how readily the caller takes the floor. Each call adds a small seeded variation, so
repeats of one scenario differ. During the call, a refusal or a repeated agent turn raises pressure and
real progress eases it; frustration shows in the words and, where the voice supports it, in the voice's
emotion. A driven caller takes the floor sooner after the agent pauses.

## Interruptions

The caller can be interrupted by the agent, and its own overlaps are short (about the end of an agent's
turn). Long talk-over is not something the caller does reliably, so barge-in is not generated as a
scenario. A hang-up is refused while the agent is still speaking.

## What a caller cannot produce

One clean synthesised voice over one ambience bed. Never: slurred, garbled or cut-off speech, a second
speaker, a dropped line, keypad input the channel does not list, or anything needing a screen or a link.
Scenarios that need these are refused and should not be planned.
