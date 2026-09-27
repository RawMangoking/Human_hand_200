#!/bin/bash
# Chaos test suite — Phase 6
# Run from ~/projects/swarm-fleet/k8s
# Requires: fleet-manager StatefulSet, zenoh-router, robot-a, robot-b all already running

set -uo pipefail

RESULTS_DIR=~/projects/swarm-fleet/chaos-test-results
mkdir -p "$RESULTS_DIR"
TS=$(date +%Y%m%d-%H%M%S)

echo "=================================================="
echo "  Chaos test suite — $(date)"
echo "  Results: $RESULTS_DIR"
echo "=================================================="

find_leader() {
    for i in 0 1 2; do
        LAST=$(kubectl logs "fleet-manager-$i" 2>/dev/null | grep "Leadership status" | tail -1)
        if echo "$LAST" | grep -q "LEADER"; then
            echo "fleet-manager-$i"
            return
        fi
    done
    echo ""
}

# ==================================================
# TEST 1: Leader failover
# ==================================================
echo ""
echo "=== TEST 1: fleet-manager leader failover ==="

LEADER=$(find_leader)
if [ -z "$LEADER" ]; then
    echo "ERROR: no current leader found. Is fleet-manager running? Skipping test 1."
else
    echo "Current leader: $LEADER"
    kubectl logs "$LEADER" > "$RESULTS_DIR/t1-leader-before-$TS.log"
    kubectl logs robot-a > "$RESULTS_DIR/t1-robot-a-before-$TS.log" 2>/dev/null
    kubectl logs robot-b > "$RESULTS_DIR/t1-robot-b-before-$TS.log" 2>/dev/null

    KILL_TIME=$(date +%s)
    echo "Killing $LEADER at $(date)..."
    kubectl delete pod "$LEADER" --grace-period=0 --force >/dev/null 2>&1

    echo "Watching for new leader (up to 30s)..."
    NEW_LEADER=""
    for i in $(seq 1 30); do
        sleep 1
        for r in fleet-manager-0 fleet-manager-1 fleet-manager-2; do
            [ "$r" == "$LEADER" ] && continue
            LAST=$(kubectl logs "$r" 2>/dev/null | grep "Leadership status" | tail -1)
            if echo "$LAST" | grep -q "LEADER"; then
                NEW_LEADER=$r
                break 2
            fi
        done
    done

    FAILOVER_SECONDS=$(( $(date +%s) - KILL_TIME ))

    if [ -n "$NEW_LEADER" ]; then
        echo "RESULT: new leader elected: $NEW_LEADER (failover ~${FAILOVER_SECONDS}s)"
    else
        echo "RESULT: WARNING — no new leader detected within 30s"
    fi

    kubectl logs robot-a > "$RESULTS_DIR/t1-robot-a-after-$TS.log" 2>/dev/null
    kubectl logs robot-b > "$RESULTS_DIR/t1-robot-b-after-$TS.log" 2>/dev/null

    TASKS_DURING_GAP=$(grep -c "Received task\|WON\|completed" "$RESULTS_DIR/t1-robot-a-after-$TS.log" "$RESULTS_DIR/t1-robot-b-after-$TS.log" 2>/dev/null | awk -F: '{s+=$2} END{print s}')

    {
        echo "TEST 1: Leader failover"
        echo "Leader killed: $LEADER"
        echo "Kill time: $(date -d @"$KILL_TIME")"
        echo "New leader: ${NEW_LEADER:-NONE DETECTED}"
        echo "Failover time: ${FAILOVER_SECONDS}s"
        echo "Robot task-related log lines after kill: ${TASKS_DURING_GAP:-0}"
        echo "(nonzero means robots kept working during/after the gap)"
    } > "$RESULTS_DIR/t1-summary-$TS.txt"
    cat "$RESULTS_DIR/t1-summary-$TS.txt"

    echo ""
    echo "Waiting 10s for StatefulSet to recreate $LEADER..."
    sleep 10
    kubectl get pods | grep fleet-manager
fi

# ==================================================
# TEST 2: Robot pod failure mid-task
# ==================================================
echo ""
echo "=== TEST 2: robot-a killed mid-task ==="
echo "Waiting up to 60s for robot-a to be busy..."

BUSY=false
for i in $(seq 1 60); do
    LAST=$(kubectl logs robot-a --tail=5 2>/dev/null | tail -1)
    if echo "$LAST" | grep -qE "WON|is busy"; then
        BUSY=true
        break
    fi
    sleep 1
done

if [ "$BUSY" = true ]; then
    echo "robot-a appears busy. Recording state and killing it..."
    kubectl logs robot-a > "$RESULTS_DIR/t2-robot-a-before-crash-$TS.log"

    KILL_TIME2=$(date +%s)
    kubectl delete pod robot-a --grace-period=0 --force >/dev/null 2>&1
    echo "Killed robot-a at $(date)"

    echo "Waiting 20s to observe what happens to its in-progress task..."
    sleep 20

    kubectl logs robot-b > "$RESULTS_DIR/t2-robot-b-after-$TS.log" 2>/dev/null

    {
        echo "TEST 2: Robot pod failure mid-task"
        echo "robot-a killed at: $(date -d @"$KILL_TIME2")"
        echo "Known limitation (Phase 1 finding): no retry mechanism exists."
        echo "Check t2-robot-b-after-$TS.log manually — did robot-b pick up"
        echo "the dropped task, or did it silently vanish?"
    } > "$RESULTS_DIR/t2-summary-$TS.txt"
    cat "$RESULTS_DIR/t2-summary-$TS.txt"
else
    echo "robot-a never appeared busy within 60s. Skipping test 2 — rerun later."
fi

echo ""
echo "Restoring robot-a..."
kubectl apply -f robot-a.yaml >/dev/null 2>&1
sleep 5
kubectl get pods

echo ""
echo "=================================================="
echo "  All chaos tests complete."
echo "  Evidence saved in: $RESULTS_DIR"
echo "=================================================="
ls -la "$RESULTS_DIR"
