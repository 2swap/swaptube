SESSIONNAME="swaptube"
tmux has-session -t $SESSIONNAME &> /dev/null

if [ $? != 0 ] 
 then
    tmux new-session -s $SESSIONNAME -d
    tmux send-keys "./go.sh Solutions"
    tmux split-window -h
    tmux send-keys "bash -c 'cd src'" C-m
    tmux send-keys 'vim src/projects/Solutions.cpp'
fi

tmux attach -t $SESSIONNAME
