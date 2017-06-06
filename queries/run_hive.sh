#!/bin/bash

#run queries for analysis of grid search validation data (xxx)

SSH_USER=faezeh.salehi
SSH_SERVER=support-tools-dmz-e1a-001.caffeine.io
HOME_DIR=/home/faezeh.salehi/AdRelModel

scp *.sql $SSH_USER@$SSH_SERVER:$HOME_DIR/

echo impressions_and_categories query
ssh $SSH_USER@$SSH_SERVER \
  "hive --silent -f $HOME_DIR/impressions_and_categories.sql" > "../data/impressions_and_categories.csv"
