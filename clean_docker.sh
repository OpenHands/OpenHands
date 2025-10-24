docker images 'ghcr.io/all-hands-ai/runtime' -q | sort -u | \
while read -r id; do
  if [ -z "$(docker ps -aq --filter ancestor="$id")" ]; then
    tags=$(docker image inspect "$id" --format '{{join .RepoTags " "}}' 2>/dev/null || true)
    [ -n "$tags" ] && docker rmi $tags
    docker rmi "$id" 2>/dev/null || true
  fi
done
docker image prune -f; docker builder prune -f; docker volume prune -f

docker system prune -a -f --volumes;
