## About
This blog is built using jekyll and powered by Github Pages.

## Setup & usage
The theme used for the blog is [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy). Fork this repository and enable github pages from the repo settings. Customize the theme and you are ready to go.

### Options
For customization, you can navigate to `_config.yml` and change the variables as per your requirement. 

### Branch
- ``master``: is for deploying your blogs.
- ``chirpy``: is for writing blogs locally and merging them to ``master``

### Running Locally
- Clone the repository
- Install Ruby, gem, jekyll, and bundle using the following commands
`
  echo "-----Installing Ruby and gem-----"
  sudo apt-get -y install ruby-full build-essential zlib1g-dev

  sudo echo '# Install Ruby Gems to ~/gems' >> ~/.zshrc
  sudo echo 'export GEM_HOME="$HOME/gems"' >> ~/.zshrc
  sudo echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.zshrc
  source ~/.zshrc


  echo "-----Installing jekyll-----"
  sudo gem install jekyll bundler

  # Also, install jekyll-paginate for /blog
  echo "-----Adding additional gem requirements-----"
  sudo gem install jekyll-paginate% 
`
- Run `bundle` to install required gems and plugins
- Run `bundle exec jekyll serve` to start your local server

Last but not the least: the [Jekyll documentation](http://jekyllrb.com) is the best starting point!
