# File Download Specs

### FD-001: Download the selected workspace file

- [x] The Files toolbar offers a labeled Download action alongside Open in new window whenever a file is selected.
- [x] Downloads retrieve original bytes using the typed file client scoped to the conversation runtime. Preview transformations are never saved as the original file.
- [x] The saved filename is the selected path's basename, including spaces and Unicode characters.
- [x] Downloads work independently of whether a preview is available. The action is disabled until the selected conversation's workspace is known and while a download is pending.
- [x] Changing file selection during a request does not change the requested file or saved filename.
- [x] A failed request shows a translated error and allows retry without saving an error response as a file.
- [x] Local and Cloud conversations use their own workspace and runtime credentials. An unavailable Cloud runtime never falls back to a local backend.

This covers individual files. Directory archives and bulk downloads are outside this change.
