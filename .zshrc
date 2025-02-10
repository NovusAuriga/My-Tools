# If you come from bash you might have to change your $PATH.
# export PATH=$HOME/bin:/usr/local/bin:$PATH

# Path to your oh-my-zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# Set name of the theme to load --- if set to "random", it will
# load a random theme each time oh-my-zsh is loaded, in which case,
# to know which specific one was loaded, run: echo $RANDOM_THEME
# See https://github.com/ohmyzsh/ohmyzsh/wiki/Themes
ZSH_THEME="robbyrussell"

plugins=(git)

source $ZSH/oh-my-zsh.sh

export MANPAGER='nvim +Man!'

# User configuration

source /usr/share/zsh/plugins/zsh-sudo/zsh-sudo.zsh
source /usr/share/zsh/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
source /usr/share/zsh/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh

alias bat='batcat -l python --style=plain -pp'
alias box='mkdir -p Content Exploit Network/Scan'
alias film='python3 -m pyftpdlib -d /home/n/Downloads'
alias web='sshpass -p z ssh z@192.168.1.24'

# export MANPATH="/usr/local/man:$MANPATH"

# You may need to manually set your language environment
# export LANG=en_US.UTF-8

# Preferred editor for local and remote sessions
# if [[ -n $SSH_CONNECTION ]]; then
#   export EDITOR='vim'
# else
#   export EDITOR='mvim'
# fi

# Compilation flags
# export ARCHFLAGS="-arch x86_64"

# Set personal aliases, overriding those provided by oh-my-zsh libs,
# plugins, and themes. Aliases can be placed here, though oh-my-zsh
# users are encouraged to define aliases within the ZSH_CUSTOM folder.
# For a full list of active aliases, run `alias`.
#
# Example aliases
# alias zshconfig="mate ~/.zshrc"
# alias ohmyzsh="mate ~/.oh-my-zsh"
alias vpn="/home/n/Downloads/./clash-verge_1.5.8_amd64.AppImage"
alias xclip="xclip -selection clipboard"

fpath+=~/.zfunc; autoload -Uz compinit; compinit

# Running ollama with gpu

export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=f16       # Corrected from q4_k to f16
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_MAX_LOADED_MODELS=1
export ROCM_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION="11.0.0"

export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Running LLM Training
export CUDA_VISIBLE_DEVICES=0
