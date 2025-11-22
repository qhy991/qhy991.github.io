source "https://rubygems.org"

# GitHub Pages compatible gem
gem "github-pages", group: :jekyll_plugins

# If you want to test locally, you can use this:
# gem "jekyll", "~> 3.9.0"
# gem "minima", "~> 2.5"

group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-sitemap"
  gem "jekyll-gist"
  gem "jekyll-paginate"
  gem "jemoji"
  gem "jekyll-remote-theme"
end

# Windows does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
install_if -> { RUBY_PLATFORM =~ %r!mingw|mswin|java! } do
  gem "tzinfo-data", platforms: [:mingw, :mswin, :x64_mingw, :jruby]
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.0", :install_if => Gem.win_platform?

# kramdown v2 ships without the gfm parser by default. If you're using
# kramdown v1, comment out this line.
gem "kramdown-parser-gfm"