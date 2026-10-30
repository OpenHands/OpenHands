// The maintenance baseline recorded in the map index
// (references/feature-map/README.md), on a line of the form
//   Maintenance baseline: main@<sha> (<YYYY-MM-DD>). <prose>
// A pass reads it as BASE; a pass that changes the map rewrites the SHA and
// date to its TARGET in its PR, and merging the PR accepts the new baseline.
const LINE =
  /^Maintenance baseline: main@([0-9a-f]{7,40}) \((\d{4}-\d{2}-\d{2})\)(.*)$/m;

/** Every line that starts like the baseline line, well-formed or not. */
export function baselineLines(indexText) {
  return indexText.match(/^Maintenance baseline:.*$/gm) ?? [];
}

/** @returns {{ sha: string, date: string } | null} */
export function parseBaseline(indexText) {
  const m = LINE.exec(indexText);
  return m ? { sha: m[1], date: m[2] } : null;
}

/** The index text with the baseline line moved to `sha` and `date`. */
export function withBaseline(indexText, sha, date) {
  if (!/^[0-9a-f]{40}$/.test(sha))
    throw new Error(`baseline sha must be a full 40-hex SHA, got ${sha}`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date))
    throw new Error(`baseline date must be YYYY-MM-DD, got ${date}`);
  if (!LINE.test(indexText))
    throw new Error(
      "the map index has no `Maintenance baseline: main@<sha> (<date>)` line",
    );
  // A merge can leave two lines; moving one of them would hide the other.
  if (baselineLines(indexText).length > 1)
    throw new Error(
      "the map index has more than one `Maintenance baseline:` line; keep one",
    );
  return indexText.replace(
    LINE,
    (_, _sha, _date, rest) =>
      `Maintenance baseline: main@${sha} (${date})${rest}`,
  );
}
