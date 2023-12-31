#!/bin/bash
if [ -z $1 ]
then
    echo "Usage: . ./gitup.sh [comment] [branch]"
    echo "You must input comment"
    return
fi

if [ -z $2 ]
then
    echo "Usage: . ./gitup.sh [comment] [branch]"
    echo "You must input branch"
    return
fi

git pull $2 main
git init
git add -A
git commit -m $1
git remote rm $2
git remote add $2 git@github.com:nex-finger/WANN.git
git push $2

echo "Pushed contents in $(pwd) to $2 branch for git@github.com:nex-finger/WANN.git"
echo "-- history -----------------------------"
echo "git init"
echo "git add -A"
echo "git commit -m $1"
echo "git remote rm $2"
echo "git remote add $2 git@github.com:nex-finger/WANN.git"
echo "git push $2"

return
