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

PAPER_TITLE = "Latent Self-Consistency for Reliable Majority-Set Selection in Short- and Long-Answer Reasoning"
LINKEDIN = "https://www.linkedin.com/in/%EC%A0%95%EC%84%9D-%EC%98%A4-438528255"
HOMES = { "index.html" => "en", "ko/index.html" => "ko" }.freeze

class HomeTest < Minitest::Test
  def test_publication_in_both_languages
    HOMES.each do |path, lang|
      pubs = page(path).css("article.pub")
      assert_equal 1, pubs.size, path
      pub = pubs.first
      assert_equal PAPER_TITLE, pub.at(".pub-title").text.strip, path
      assert_equal "AAAI 2026", pub.at(".pub-venue").text.strip, path
      assert_equal "Jungsuk Oh", pub.at(".pub-authors strong.me").text.strip, path
      hrefs = pub.css("a").map { |a| a["href"] }
      assert_includes hrefs, "https://arxiv.org/abs/2508.18395", path
      assert_includes hrefs, "https://github.com/jeongseokO/LatentSC", path
      assert_includes hrefs, "https://huggingface.co/jeongseokoh", path
      assert_includes hrefs, (lang == "en" ? "/research/lsc/" : "/ko/research/lsc/"), path
      assert_includes pub.at("pre").text, "@inproceedings{oh2026latent", path
    end
  end

  def test_profile_links
    HOMES.each_key do |path|
      hrefs = page(path).css("a").map { |a| a["href"] }
      [
        "mailto:luke0112@snu.ac.kr",
        "https://github.com/jeongseokO",
        "https://huggingface.co/jeongseokoh",
        "https://scholar.google.com/citations?user=WV5svAIAAAAJ",
        LINKEDIN,
      ].each { |url| assert_includes hrefs, url, path }
      raw = hrefs.compact.grep(/[^\x00-\x7F]/)
      assert_empty raw, "#{path}: non-ASCII href #{raw.inspect}"
    end
  end

  def test_advisor_and_lab_links
    HOMES.each_key do |path|
      hrefs = page(path).css("a").map { |a| a["href"] }
      assert_includes hrefs, "https://leejayyoon.github.io/", path
      assert_includes hrefs, "https://www.skiml.snu.ac.kr/", path
    end
  end

  def test_nav_anchors_resolve_from_every_page
    ids = %w[about research publications]
    HOMES.each_key do |path|
      doc = page(path)
      assert_equal ids.map { |i| "##{i}" }, doc.css(".site-nav a").map { |a| a["href"] }, path
      ids.each { |i| refute_nil doc.at_css("##{i}"), "#{path}: missing ##{i}" }
    end
    { "research/lsc/index.html" => "/", "ko/research/lsc/index.html" => "/ko/" }.each do |path, base|
      assert_equal ids.map { |i| "#{base}##{i}" }, page(path).css(".site-nav a").map { |a| a["href"] }, path
    end
  end

  def test_interests_and_education
    en = page("index.html")
    assert_equal ["Efficient LLM", "Trustworthy AI", "Prompt Engineering"],
                 en.css(".interest h3").map { |h| h.text.strip }
    assert_equal 3, en.css(".edu").size
    assert_includes en.at("#education").text, "Hanyang University"
    ko = page("ko/index.html")
    assert_equal 3, ko.css(".interest").size
    assert_equal 3, ko.css(".edu").size
    assert_includes ko.at("#education").text, "한양대학교"
  end
end
