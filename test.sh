#!/bin/bash

# List of demo projects
mapfile -t DEMOS < <(find src/Projects/Demos -type f -name '*.cpp' -printf '%f\n' | sed 's/\.cpp$//')

PASS_COUNT=0
FAIL_COUNT=0
FAILED_PROJECTS=()

echo "===================================================="
echo "                SWAPTUBE SMOKETESTS"
echo "===================================================="

for demo in "${DEMOS[@]}"; do
    echo ""
    echo "---- Running smoketest for project: ${demo} ----"
    ./go.sh "$demo" 160 90 30 -s > /dev/null
    if [ $? -eq 0 ]; then
        echo "✅ ${demo}: PASS"
        PASS_COUNT=$((PASS_COUNT+1))
    else
        echo "❌ ${demo}: FAIL"
        FAIL_COUNT=$((FAIL_COUNT+1))
        FAILED_PROJECTS+=("$demo")
    fi
done

echo ""
echo "===================================================="
echo "SUMMARY"
echo "===================================================="
echo "Passed: ${PASS_COUNT}"
echo "Failed: ${FAIL_COUNT}"

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "Failed projects:"
    for f in "${FAILED_PROJECTS[@]}"; do
        echo "  - $f"
    done
    exit 1
else
    echo "All smoketests passed!"
    exit 0
fi
