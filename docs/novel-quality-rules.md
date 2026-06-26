# Novel Quality Rules

This document defines reusable quality rules for long-form fiction generation.
Use it as a shared prompt module for outline generation, chapter drafting,
revision, and final quality review.

## Core Objective

Produce chapters that feel intentional, vivid, and internally consistent. The
reader should always understand:

- what changed in the scene
- what the protagonist wants
- what pressure blocks that desire
- why the next scene is necessary

## Drafting Standards

1. Every chapter needs a clear dramatic question.
   Avoid chapters that only explain background, travel, cultivation, logistics,
   or relationship status. If a chapter does not create, escalate, reverse, or
   resolve a conflict, revise it before publishing.

2. Open with motion, pressure, or a charged image.
   Do not begin with generic weather, abstract emotion, encyclopedia exposition,
   or a summary of previous events unless the recap is tied to immediate action.

3. Keep cause and effect visible.
   Each important action should create a consequence. Each consequence should
   force a choice, reveal character, or change the available options.

4. Make conflict specific.
   Replace vague obstacles such as "danger", "trouble", or "misunderstanding"
   with concrete stakes: a secret may be exposed, a debt comes due, a promise is
   broken, a rival gains leverage, a resource is lost, or a deadline closes in.

5. Preserve character intent.
   Before writing a scene, identify each major character's visible goal, hidden
   motive, and emotional limit. Dialogue and action should reflect those goals
   instead of serving only the plot outline.

6. Show emotion through behavior.
   Prefer gesture, rhythm, avoidance, silence, contradiction, and choice over
   direct labels such as "angry", "sad", "shocked", or "moved".

7. Balance scene and summary.
   Important turning points should be dramatized with sensory detail, dialogue,
   and decisions. Routine transitions can be summarized briefly.

8. Avoid repetitive sentence rhythm.
   Mix short impact lines with longer descriptive or reflective sentences. Watch
   for repeated openings, repeated emotional beats, and repeated paragraph
   lengths.

9. Cut filler.
   Remove empty emphasis, redundant inner monologue, circular dialogue, generic
   praise, over-explained reactions, and repeated descriptions of the same power,
   beauty, status, or pain.

10. End with forward pull.
    A chapter ending should introduce a new complication, reveal new information,
    lock in a decision, sharpen a dilemma, or make the reader fear the next cost.

## Continuity Rules

- Track names, titles, relationships, locations, timelines, abilities, injuries,
  possessions, promises, secrets, and unresolved conflicts.
- Do not introduce a convenient ability, clue, ally, or coincidence unless it was
  seeded earlier or immediately costs something meaningful.
- Do not resolve major tension through luck, sudden off-screen information, or a
  character acting against established motivation.
- When revisiting a location or relationship, show what has changed since the
  previous appearance.

## Style Rules

- Use concrete nouns and active verbs.
- Keep metaphors fresh, local to the scene, and limited in number.
- Avoid modern internet phrasing unless the story setting supports it.
- Avoid tonal drift. Humor, romance, suspense, horror, and grandeur should match
  the genre promise of the book.
- Avoid excessive lists of adjectives. One precise detail is usually stronger
  than three generic ones.
- Let subtext carry dialogue. Characters should not always say exactly what they
  mean.

## Chapter Review Checklist

Before accepting a generated chapter, score each item from 1 to 5.

| Item | Question |
| --- | --- |
| Hook | Does the opening create immediate curiosity or tension? |
| Goal | Does the protagonist want something specific in this chapter? |
| Conflict | Is there active resistance, not just description? |
| Change | Is the situation meaningfully different by the end? |
| Character | Do decisions reveal personality, values, or flaws? |
| Continuity | Does the chapter respect established facts and promises? |
| Texture | Are there concrete sensory details and grounded actions? |
| Dialogue | Does dialogue contain intent, friction, or subtext? |
| Pacing | Are exposition, action, reflection, and dialogue balanced? |
| Ending | Does the ending create a reason to keep reading? |

Reject or revise any chapter with:

- total score below 38
- any item scored 2 or lower
- contradiction with established canon
- repeated scene structure from the previous chapter
- more than two paragraphs in a row without new action, decision, or discovery

## Draft Prompt Module

Use this module inside the chapter-writing prompt.

```text
Write the chapter as a polished scene-driven novel chapter, not as an outline or
summary.

Quality requirements:
- Give the chapter one clear dramatic question.
- Start with immediate pressure, action, or a charged image.
- Make the protagonist's goal concrete within the first third of the chapter.
- Escalate through cause and effect; every major action must create a consequence.
- Show emotion through behavior, dialogue rhythm, and choices instead of labels.
- Use specific sensory details, but avoid decorative description that does not
  affect mood, conflict, or character.
- Keep dialogue purposeful: each exchange should reveal motive, shift power, hide
  information, or force a decision.
- Preserve continuity with established names, relationships, timeline, abilities,
  injuries, secrets, and unresolved promises.
- Do not resolve conflict by coincidence, sudden unseeded power, or off-screen
  convenience.
- End with a changed situation and forward pull.

Avoid:
- generic exposition dumps
- repetitive inner monologue
- repeated sentence openings
- empty reactions such as "everyone was shocked"
- overusing status, beauty, aura, killing intent, silence, or vague danger
- summarizing key confrontations instead of dramatizing them
```

## Revision Prompt Module

Use this module after a draft is generated.

```text
Revise the chapter for publishable fiction quality.

Keep the plot events and canon facts intact, but improve:
- opening hook
- scene goal and conflict
- cause-and-effect progression
- concrete sensory grounding
- character-specific dialogue
- emotional subtext
- sentence rhythm
- chapter ending

Remove or rewrite:
- filler paragraphs
- repeated reactions
- vague emotional labels
- redundant exposition
- contradictions with established continuity
- convenient resolutions that were not earned

Return only the revised chapter text.
```

## Quality Judge Prompt Module

Use this module as a separate review pass. The judge should not rewrite unless
asked; it should diagnose problems precisely.

```text
Evaluate the chapter against the novel quality checklist.

Return:
1. Scores from 1 to 5 for Hook, Goal, Conflict, Change, Character, Continuity,
   Texture, Dialogue, Pacing, and Ending.
2. The three highest-impact problems, each with a concrete example.
3. Whether the chapter should be ACCEPTED, REVISED, or REJECTED.
4. If revised, provide a concise revision brief with no more than five actions.

Reject if any score is 2 or lower, if total score is below 38, if continuity is
broken, or if the chapter repeats the previous chapter's scene structure.
```

## Recommended Generation Flow

1. Plan the chapter with dramatic question, protagonist goal, opposition,
   turning point, and ending hook.
2. Draft the chapter with the Draft Prompt Module.
3. Review with the Quality Judge Prompt Module.
4. If the result is REVISED or REJECTED, revise with the Revision Prompt Module.
5. Run the judge again.
6. Accept only when all hard rejection rules pass.

