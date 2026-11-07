Build a factual story bible for a NEW vertical-microdrama series.
Use only supplied loglines, continuity notes, speaker samples, camera/action
context, and evidence summaries derived from those sources. Hints are OPTIONAL:
when identity_hints is empty, infer identities solely from the evidence.
Never invent surnames or titles that are not in the text. Do not infer identity
from a costume or vague resemblance alone. Compare who is addressed, who acts,
and the continuity across episodes. Keep uncertain identities separate.

For every main character return name, exact speaker aliases, role, relationships,
confidence (high/medium/low), and evidence (a short source quote or canonical line ID).
Main characters appear in at least two episodes through their aliases OR are
named in a logline. Put less important one-episode speakers in minor_characters,
a list of names only; use their exact speaker labels so coverage is verifiable.
All distinct labels must be covered by main aliases or minor_characters. Do not
hide a recurring speaker in minor_characters. Order main characters by importance.

Return title_guess (clearly tentative), a 2–3 sentence premise, main conflicts,
and episode-range arcs. No invented events, motives, names, or relationships.
If the sources are insufficient, explicitly mark uncertainty and low confidence.
Return only JSON matching the provided schema.
