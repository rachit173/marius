#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

BASE_DIR=$(pwd)
echo $BASE_DIR

dataset=$1

kill_previous_processes() {
  # Kill stat workers
  echo "-- Killing DSTAT on host."
  {
    ps -ef | grep dstat | awk {'print$2'} | xargs kill
  } || {
    echo "-- Killing DSTAT on host gives erorr."
  }

  # kill previous coordinator processes
  echo "-- Killing coordinator on host gives error."
  {
    ps -ef | grep coordinator.sh | awk {'print$2'} | xargs kill
  } || {
    echo "-- Killing coordinator on host gives error."
  }

  # kill previous worker processes
  echo "-- Killing worker on host gives error."
  {
    ps -ef | grep worker.sh | awk {'print$2'} | xargs kill
  } || {
    echo "-- Killing worker on host gives error."
  }
}

start_logging() {
  echo "-- Creating directories for logs"
  STATS_DIR=${BASE_DIR}/stats
  TS=$(date +%s)
  DSTAT_FILE=$STATS_DIR/marius_train_${dataset}_dstat_${TS}.csv
  echo "-- Writing DSTAT logs at $DSTAT_FILE"
  mkdir -p $STATS_DIR
  dstat -cdngyimrtpsy --fs --nvidia-gpu --output $DSTAT_FILE 1 > /dev/null &
  dstat_pid=$!
}

stop_logging() {
  echo "--Stopping logging process"
  sleep 3
  kill -9 $dstat_pid
}

run() {
   marius_train examples/training/configs/$dataset.ini
}

main() {
  kill_previous_processes
  start_logging
  run
  stop_logging
}

main
