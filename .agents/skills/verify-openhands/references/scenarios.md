# Scenarios a happy path never shows

A family can be fully mapped, every bullet passing, and a user still reports a
bug on it the same week. Reading the fixes other people sent after this map
existed, the page was almost always mapped already. What was missing was the
*situation*: the user came back to the thing a day later, did two things at
once, had real data instead of a two-row fixture, or ran it somewhere other
than a fresh local stack.

This file lists those situations as questions to ask of a family. Use it in
three places:

- **Mapping** ([mapping.md](mapping.md) §3): after the happy path of a
  sub-feature is written, walk the list and add a bullet for each question
  that has an answer on this surface.
- **The daily pass** ([daily.md](daily.md) tier 2): for each merged change,
  ask the questions of the surface it touched, not only whether the change
  does what its PR says.
- **A full pass** ([maintenance.md](maintenance.md) §3): each source reader
  returns, with its drift list, the questions its family has no bullet for.

A question with no answer on a surface is skipped, not recorded. A question
whose answer needs something the run lacks is `blocked` with the prerequisite,
as everywhere else. Every bullet added this way is driven live before it is
written, like any other.

## 1. Was every control operated?

`control-openhands map controls` lists the interactive elements in `src/` whose
test id no map file mentions (`--family Fnn` for one family, `--base
<baseline>` for the controls in files that changed since the last pass). `map
coverage` cannot see these: it counts a page as mapped once any part of it is.

For each control, the bullet says what a click does, then asks twice more:

- What does it do the **second** time (reopen the menu, toggle back, submit
  again)?
- What does it do when its list is **populated** and when it is **empty**?

A control named only in the family's description ("the form has Add Filter and
Ask Agent") has not been operated. A report of the form "the button does
nothing" always points at one of these.

## 2. Did the user come back to it?

Most recipes create, check and delete. Users create, leave and return.

- **Edit what exists.** Open the edit form of an item that was saved earlier
  with every optional field filled (repositories, a script bundle, a stored
  credential, a non-default profile). Change one unrelated field, save, and
  read *every* field back. A field the form did not load is lost on save.
- **Round trips.** Export then import, duplicate, rename, move: read the
  result back field by field against the original.
- **Reopen after it is over.** Whatever a conversation showed while it ran
  (terminal output, a Thinking row and its reasoning, hook events, the title,
  timestamps, the order of events) is read again after `browser reload`,
  again in a `browser tab new`, and again after `control-openhands restart`.
  Live rendering and rendering from saved history are two code paths; a
  bullet that only watched the live one proves half.
- **The next day.** Order and age labels are checked with events that are not
  from this minute: `browser clock --system +86400000` before the `goto`, and
  `browser timezone <IANA name>` for a browser that is not in the server's
  zone.

## 3. Did the user do something else meanwhile?

A recipe waits for each step to settle. Users do not.

- While a conversation is **starting**: navigate to another conversation,
  send a second message, attach a file, open Settings and come back.
- While a save or an upload is **in flight**: click again, press Escape,
  switch tabs in the panel, leave the page.
- While the agent is **running**: switch conversations and back, reload,
  open the same conversation in a second tab.

Hold the pending state open instead of racing it: `control-openhands service
pause agent-server` (or `automation`) queues real requests until `service
resume`; `browser network-state offline` cuts the browser's side. Read what
happened with `browser network --order completion` and `browser toasts
--history`. The result to assert is about the *other* place too: focus stays
where the user went, the draft they typed is still there, nothing from the
conversation they left shows up in the one they opened.

## 4. What did it look like on the way?

An end state can be right after a wrong transition: a toast that flashes for
a fast start, "Thinking" shown while history replays, a title that fades out
and in, a row that jumps, the same error twice.

- Record the transition, not only the end: `browser click ... --observe SEL`,
  `browser scroll --observe SEL`, `browser record` for layout, and a counted
  `browser toasts --history` (cleared first) for anything that must appear
  once.
- Judge the transition as a user would. Something that breaks the
  experience on the way (a flash, a jump, a delay, a duplicate) is a product
  defect even when the end state is right: file it, and state in the bullet
  the behavior a user would want, with a `Known failure` note as
  [maintenance.md](maintenance.md) §5 describes.

## 5. Was the data like a user's?

Fixtures are short, few and ASCII. Give each list, table, name and diff one
bullet with content that is not:

- **Long**: a table cell with a paragraph in it, a repository or branch name
  past the width of its picker, a single unbroken token.
- **Many**: past the first page of whatever paginates (more than 20
  conversations, more events than one history page, two events with the same
  timestamp at the page boundary), a diff large enough to time.
- **Odd**: names that are percent-encoded, non-ASCII or typed through an IME
  composition; two items with the same name from different sources; a file
  made of repeated lines.
- **From another provider**: every place that builds a link or lists
  repositories is driven with each remote shape the product supports, not
  only GitHub (`fixture git-repo --remote` takes Azure DevOps, GitLab,
  Bitbucket and Bitbucket Data Center URLs).

## 6. Was it run where users run it?

`launch` gives a fresh local stack at `/` on a loopback origin, in the
machine's time zone, with free ports. Each of those is a choice a user may not
share. Where a family's behavior can depend on one, it gets a bullet, or a
`blocked` row naming exactly what is missing:

- A **base path**, with and without the trailing slash, and a hostname that
  happens to contain the mount path.
- A **plain-HTTP origin that is not loopback** (browser APIs for secure
  contexts are absent there).
- A browser **time zone** other than the server's (`browser timezone`), and
  a server host that is not in UTC (`TZ=<zone> control-openhands launch
  --new`; the launcher hands `TZ` to the services).
- A **taken port**, a service that **dies** after start (`service stop`), a
  session key that **rotates** (`restart --rotate-key`).
- **Docker**: state that must survive the container being recreated, not
  only restarted. **Windows** and the **desktop app**: paths, modifier keys,
  window chrome.
- **Cloud and Enterprise backends**: a paused sandbox, a sandbox that takes
  seconds to start, a member who is not an admin, an instance that switched a
  feature off. This is the largest group of reports and the one a local run
  reaches least. A row that is `blocked` on a Cloud account was not checked:
  the pass report counts those rows separately, so the gap stays visible
  instead of reading as coverage.

## Illustrative, not normative, not exclusive

These six are illustrative, not normative, not exclusive. Use your judgment
to only apply them when necessary, when potential bugs would impair the user
experience, not theoretical. They come from experience with the map since
its first version (2026-10-06). When a new kind of report does not fit any
question here, you can add the question.
