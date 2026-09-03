# SVG-edit Canvas Extension example

This installable example exercises the Canvas Extension page ABI with
[SVG-edit 7.4.2](https://github.com/SVG-Edit/svgedit/releases/tag/v7.4.2).
It uses the published SVG-edit application in an iframe so the editor's global
styles, element IDs, and keyboard handlers remain isolated from Agent Canvas.

Install this directory through **Customize -> Extensions** using its absolute
path on the Agent Server machine. Enable **SVG Editor**, then open its sidebar
item. Disabling or uninstalling the extension removes the route and iframe.

This first slice demonstrates page installation and lifecycle only. Passing a
workspace SVG into the editor and saving it back require a narrow host-to-frame
file bridge and are intentionally left to the follow-up integration.
