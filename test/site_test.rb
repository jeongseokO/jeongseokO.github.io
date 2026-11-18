# Checks the built site in _site/. Run through script/test, which builds first.
require "minitest/autorun"
require "nokogiri"

SITE = File.expand_path("../_site", __dir__)
BASE = "https://jeongseoko.github.io"

# path in _site => [lang, public URL, counterpart URL]
PAGES = {
  "index.html"                 => ["en", "/",                 "/ko/"],
  "ko/index.html"              => ["ko", "/ko/",              "/"],
  "research/lsc/index.html"    => ["en", "/research/lsc/",    "/ko/research/lsc/"],
  "ko/research/lsc/index.html" => ["ko", "/ko/research/lsc/", "/research/lsc/"],
}.freeze

def page(path)
  file = File.join(SITE, path)
  raise "missing #{path} in _site" unless File.exist?(file)
  Nokogiri::HTML(File.read(file))
end

class LanguageTest < Minitest::Test
  def test_pages_exist_with_lang
    PAGES.each do |path, (lang, _, _)|
      assert_equal lang, page(path).at("html")["lang"], path
    end
  end

  def test_language_switch_points_to_counterpart
    PAGES.each do |path, (_, _, other)|
      links = page(path).css("a.lang-switch")
      assert_equal 1, links.size, "#{path}: one language switch"
      assert_equal other, links.first["href"], path
    end
  end

  def test_hreflang_alternates
    PAGES.each do |path, (lang, url, other)|
      doc = page(path)
      en_url, ko_url = lang == "en" ? [url, other] : [other, url]
      alt = ->(code) { doc.at(%(link[rel="alternate"][hreflang="#{code}"]))&.[]("href") }
      assert_equal BASE + en_url, alt.("en"), path
      assert_equal BASE + ko_url, alt.("ko"), path
      assert_equal BASE + en_url, alt.("x-default"), path
    end
  end
end
