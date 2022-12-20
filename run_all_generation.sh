#!/bin/bash

declare -a StringArray=(
'config/Adult/config-adult-gender.yml'
'config/Adult-debiased/config-adult-gender.yml'
'config/Crime/config-crime-race.yml'
'config/German/config-german-gender.yml'
)

for conf in "${StringArray[@]}"; do
  echo ${conf}
  python3 -u mainGenerate.py ${conf}
done