for p in checkpoints*; do
  [ -e "$p" ] || continue
  echo "---- $p ----"
  ls -ldO@ "$p"
  stat -f 'name=%N type=%HT perms=%Sp owner=%Su:%Sg flags=%Sf' "$p"
done

for p in checkpoints*; do
  [ -e "$p" ] || continue
  sudo chflags -R nouchg "$p"
done

for p in checkpoints*; do
  [ -e "$p" ] || continue
  sudo chown -R "$USER":staff "$p"
  sudo chmod -R u+rwX "$p"
done

for p in checkpoints*; do
  [ -e "$p" ] || continue
  mv -v "$p" "${p}_renamed"
done