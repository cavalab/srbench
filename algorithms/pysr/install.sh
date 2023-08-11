# First, install Julia:
case "$(uname -sr)" in

   Darwin*)
     echo 'Installing Julia on Mac OS X'
     curl -fsSL https://install.julialang.org | sh -s -- -y
     ;;

   Linux*Microsoft*)
     echo 'Installing Julia on WSL'  # Windows Subsystem for Linux
     curl -fsSL https://install.julialang.org | sh -s -- -y
     ;;

   Linux*)
     echo 'Installing Julia on Linux'
     curl -fsSL https://install.julialang.org | sh -s -- -y
     ;;

   CYGWIN*|MINGW*|MINGW32*|MSYS*)
     echo 'Installing Julia on MS Windows'
     winget install julia -s msstore
     ;;

   *)
     echo 'OS not detected. Trying to use curl...'
     curl -fsSL https://install.julialang.org | sh -s -- -y
     ;;
esac
# (architecture detection tip from https://stackoverflow.com/questions/3466166/how-to-check-if-running-in-cygwin-mac-or-linux)

# Then, install Julia dependencies
python -m pysr install