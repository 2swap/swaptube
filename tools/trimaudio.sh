find . -type f -iname "*.mp3" -exec sh -c 'tempfile=$(mktemp).mp3 && sox "$1" "$tempfile" trim 0 -0.1 && mv "$tempfile" "$1"' sh {} \;
