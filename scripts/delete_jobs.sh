status=Completed

schedctl list | grep $status | awk -F '|' '{print $3}' | xargs -n1 schedctl delete