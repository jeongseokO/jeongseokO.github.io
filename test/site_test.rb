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
    assert_equal ["LLM Architecture", "Reliability", "Prompt Engineering"],
                 en.css(".pillar h3").map { |h| h.text.strip }
    assert_equal 3, en.css(".edu").size
    assert_includes en.at("#education").text, "Hanyang University"
    ko = page("ko/index.html")
    assert_equal 3, ko.css(".pillar").size
    assert_equal 3, ko.css(".edu").size
    assert_includes ko.at("#education").text, "한양대학교"
  end
end

LSC_PAGES = %w[research/lsc/index.html ko/research/lsc/index.html].freeze

class ResearchPageTest < Minitest::Test
  def test_lsc_pages_structure
    LSC_PAGES.each do |path|
      doc = page(path)
      %w[tldr problem method results citation].each do |id|
        refute_nil doc.at_css("section##{id}"), "#{path}: missing section ##{id}"
      end
      img = doc.at('img[src$="lsc-overview.svg"]')
      refute_nil img, "#{path}: overview figure"
      refute_empty img["alt"].to_s.strip, "#{path}: figure alt text"
      assert_includes doc.at("#citation pre").text, "@inproceedings{oh2026latent", path
    end
  end

  # Values from Table 1 of arXiv:2508.18395v3: short avg, long avg, time %, memory %.
  def test_lsc_results_table_matches_paper
    LSC_PAGES.each do |path|
      table = page(path).at("table.results")
      refute_nil table, path
      assert_equal 15, table.css("tbody tr").size, path
      row = ->(model, method) do
        tr = table.at(%(tbody tr[data-model="#{model}"][data-method="#{method}"]))
        refute_nil tr, "#{path}: row #{model}/#{method}"
        tr.css("td.num").map { |td| td.text.strip }
      end
      assert_equal %w[72.3 65.2 0.2 0.005], row.("LLaMA3.1-8B", "LSC"), path
      assert_equal %w[77.1 N/A], row.("Qwen3-8B", "SC").first(2), path
      assert_equal "18.2", row.("Qwen3-8B", "USC")[2], path
      assert_equal %w[82.6 77.6 0.9 0.001], row.("LLaMA3.3-70B", "LSC"), path
    end
  end
end

# Inner layouts and page bodies render before the outer layout, so UI strings must be
# available there too. These pin visible labels in both languages.
class LabelTest < Minitest::Test
  def test_home_section_labels
    { "index.html" => %w[About Publications Copy], "ko/index.html" => %w[소개 논문 복사] }.each do |path, (about, pubs, copy)|
      doc = page(path)
      assert_equal about, doc.at("#h-about").text.strip, path
      assert_equal pubs, doc.at("#h-publications").text.strip, path
      assert_equal copy, doc.at("#publications .copy-btn").text.strip, path
    end
  end

  def test_research_page_labels
    { "research/lsc/index.html" => ["/#research", "Research", "Model", "Copy"],
      "ko/research/lsc/index.html" => ["/ko/#research", "연구", "모델", "복사"] }.each do |path, (href, crumb, model, copy)|
      doc = page(path)
      link = doc.at(".eyebrow a")
      assert_equal href, link["href"], path
      assert_equal crumb, link.text.strip, path
      assert_equal model, doc.at("table.results thead th").text.strip, path
      refute_empty doc.at("table.results caption").text.strip, path
      assert_equal copy, doc.at("#citation .copy-btn").text.strip, path
    end
  end
end

# Final-review fixes.
require "tmpdir"
require "fileutils"

class ReviewFixTest < Minitest::Test
  # A future paper without project_ref must not link to some arbitrary page.
  def test_paper_without_project_page_gets_no_project_link
    Dir.mktmpdir do |dir|
      src = File.expand_path("..", __dir__)
      %w[_config.yml _data _includes _layouts assets index.md ko research 404.html].each do |f|
        FileUtils.cp_r(File.join(src, f), dir)
      end
      File.open(File.join(dir, "_data/publications.yml"), "a") do |f|
        f.write("\n- id: future\n  title: A Future Paper\n  authors:\n    - name: Jungsuk Oh\n      me: true\n  venue: Workshop 2027\n  year: 2027\n")
      end
      out = File.join(dir, "_site")
      system("bundle", "exec", "jekyll", "build", "-q", "-s", dir, "-d", out, exception: true)
      HOMES.each_key do |path|
        future = Nokogiri::HTML(File.read(File.join(out, path))).at("#pub-future")
        refute_nil future, path
        assert_nil future.at(".pub-title a"), "#{path}: title must not link anywhere"
        assert_empty future.css(".chip--accent"), "#{path}: no project-page chip"
      end
    end
  end

  def test_footer_has_contact_on_every_page
    PAGES.each_key do |path|
      hrefs = page(path).css(".site-footer a").map { |a| a["href"] }
      assert_includes hrefs, "mailto:luke0112@snu.ac.kr", path
      assert_includes hrefs, "https://scholar.google.com/citations?user=WV5svAIAAAAJ", path
    end
  end

  def test_korean_copy
    lsc = page("ko/research/lsc/index.html").at("main").text
    ["같은 수준", "일관된지", "이웃들하고만"].each { |bad| refute_includes lsc, bad }
    assert_includes lsc, "비슷한 수준"
    home = page("ko/index.html")
    assert_equal "서울대학교 데이터사이언스대학원 박사과정", home.at(".hero__role").text.strip
    assert_equal "지도교수 이재윤 · SKI-ML Lab", home.at(".hero__advisor").text.gsub(/\s+/, " ").strip
    assert_equal "한양대학교", home.css(".edu__school").last.text.strip
    home.css(".tags li").each { |li| assert_match(/\S \(.+\)\z/, li.text.strip, "Korean tag with English gloss") }
  end

  # Table 4: compare with the strongest baseline per model (WUCS, WUCS, USC).
  def test_consistency_compared_with_strongest_baseline
    assert_includes page("research/lsc/index.html").at("#results").text, "80.3%, 86.5%, and 90.1%"
    assert_includes page("ko/research/lsc/index.html").at("#results").text, "80.3%, 86.5%, 90.1%"
  end

  def test_numbers_cite_paper_sections
    LSC_PAGES.each do |path|
      text = page(path).at("main").text
      ["(Introduction)", "(Method)", "(Training Phase)", "(Experimental Setup)"].each do |cite|
        assert_includes text, cite, "#{path}: #{cite}"
      end
    end
  end
end

# Research interests are three pillars that together aim at trustworthy AI.
class ResearchStructureTest < Minitest::Test
  PILLARS = ["LLM Architecture", "Reliability", "Prompt Engineering"].freeze

  def test_each_pillar_has_an_illustration
    HOMES.each_key do |path|
      cards = page(path).css("#research .pillar")
      assert_equal 3, cards.size, path
      cards.each do |card|
        svg = card.at("svg[role=img]")
        refute_nil svg, "#{path}: pillar illustration"
        refute_empty svg.at("title").text.strip, "#{path}: illustration title"
      end
    end
  end

  def test_pillars_converge_on_trustworthy_ai
    HOMES.each_key do |path|
      fig = page(path).at("#research svg.convergence")
      refute_nil fig, "#{path}: convergence diagram"
      assert_includes fig.text, "Trustworthy AI", path
      PILLARS.each { |name| assert_includes fig.text, name, path }
    end
  end

  def test_research_terms
    HOMES.each_key do |path|
      text = page(path).at("#research").text
      refute_includes text, "Architecture pruning", path
      assert_includes text, "Structured pruning", path
      assert_includes text, "Linear & hybrid attention", path
    end
  end
end
