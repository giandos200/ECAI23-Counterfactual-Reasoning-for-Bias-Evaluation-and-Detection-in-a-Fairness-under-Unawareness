#!/bin/bash

declare -a StringArray=(
'config/Adult/config-adult-gender.yml'
'config/Adult/config-adult-MS.yml'
'config/Adult-debiased/config-adult-gender.yml'
'config/Adult-debiased/config-adult-MS.yml'
'config/Crime/config-crime-race.yml'
'config/German/config-german-gender.yml'
'config/German/config-german-age.yml'
)

for conf in "${StringArray[@]}"; do
  echo ${conf}
  python3 -u mainGenerate.py ${conf}
done