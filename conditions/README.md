# Conditions

A condition fixes everything that changes behaviour: the generator and reviewer models and their
immutable revisions, prompts, thinking mode, decoding settings, token limits, and the grader and
feature versions. Seeds live in `replicates/`, not here.

A condition may `extends` a frozen condition and list only its `overrides`. Mappings merge
recursively, lists are replaced as a whole, an explicit `null` is a value, and cycles are invalid.
Before a run the chain is resolved to `condition.resolved.yaml`, and the condition hash covers that
resolved configuration plus the bytes of every referenced file. A completed run is never edited in
place; any behavioural change creates a new condition.
