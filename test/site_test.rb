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
  "research/speed/index.html" => ["en", "/research/speed/", "/ko/research/speed/"],
  "ko/research/speed/index.html" => ["ko", "/ko/research/speed/", "/research/speed/"],
  "research/translation-inference/index.html" => ["en", "/research/translation-inference/", "/ko/research/translation-inference/"],
  "ko/research/translation-inference/index.html" => ["ko", "/ko/research/translation-inference/", "/research/translation-inference/"],
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
      pub = page(path).at("article#pub-lsc2026")
      refute_nil pub, path
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

  # The overview stays valid as papers are added, so it names no specific paper.
  def test_diagram_names_no_paper
    HOMES.each_key do |path|
      page(path).css("#research svg.convergence").each do |fig|
        refute_match(/LSC|Self-Consistency/, fig.text, path)
      end
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

# Copy should read like a person wrote it: concrete topics instead of slogans,
# no stock triads or translated-sounding connectors.
class NaturalCopyTest < Minitest::Test
  def test_diagram_shows_topics_not_slogans
    HOMES.each_key do |path|
      fig = page(path).at("#research svg.convergence").text
      ["Linear attention · Pruning", "Consistency · Hallucination", "CoT · In-context learning"].each do |k|
        assert_includes fig, k, path
      end
      ["Efficient to run and to check", "Answers that agree and hold up", "Behavior as intended",
       "실행·검증 비용을 낮춤", "일관되고 사실에 맞는 답", "의도한 대로 동작"].each do |slogan|
        refute_includes page(path).at("main").text, slogan, path
      end
    end
  end

  def test_no_stock_phrases
    en = page("index.html").at("main").text
    ["relied on, checked, and steered", "each supplies one part", "My research goal is trustworthy AI:", "My research asks"].each do |s|
      refute_includes en, s
    end
    ko = page("ko/index.html").at("main").text
    ["즉,", "이를 위해", "한 부분을 맡습니다", "믿을 수 있고, 검증 가능하며"].each { |s| refute_includes ko, s }
    { "research/lsc/index.html" => ["each gives something up", "is not a way out either"],
      "ko/research/lsc/index.html" => ["각각 대가가 있습니다", "해결책이 되지 못합니다"] }.each do |path, bad|
      text = page(path).at("main").text
      bad.each { |s| refute_includes text, s, path }
    end
  end
end

SPEED_TITLE = "Shallow Prefill, Deep Decoding: Efficient Long-Context Inference via Layer-Asymmetric KV Visibility"

class PreprintAndProjectTest < Minitest::Test
  def test_speed_preprint_listed_first_in_both_languages
    HOMES.each do |path, _lang|
      doc = page(path)
      ids = doc.css("#publications article.pub").map { |a| a["id"] }
      assert_equal %w[pub-speed2026 pub-lsc2026], ids, path
      pub = doc.at("#pub-speed2026")
      assert_equal SPEED_TITLE, pub.at(".pub-title").text.strip, path
      assert_equal "Preprint", pub.at(".pub-venue").text.strip, path
      assert_equal "Jungsuk Oh", pub.at(".pub-authors strong.me").text.strip, path
      %w[Hyeseo\ Jeon Hyunjune\ Ji Kyongmin\ Kong Jay-Yoon\ Lee].each do |name|
        assert_includes pub.at(".pub-authors").text, name.delete("\\"), path
      end
      assert_includes pub.css("a").map { |a| a["href"] }, "https://arxiv.org/abs/2605.06105", path
      project_href = path.start_with?("ko/") ? "/ko/research/speed/" : "/research/speed/"
      assert_equal [project_href], pub.css(".chip--accent").map { |a| a["href"] }, path
      assert_includes pub.at("pre").text, "@article{oh2026shallow", path
    end
  end

  def test_corresponding_author_note_once
    HOMES.each_key do |path|
      assert_equal 1, page(path).css("#publications .pub-note").size, path
    end
  end

  def test_mfds_project_in_both_languages
    { "index.html" => ["Ministry of Food and Drug Safety", "Feb 2025 – Nov 2026"],
      "ko/index.html" => ["식품의약품안전처", "2025.02 – 2026.11"] }.each do |path, (org, period)|
      projects = page(path).css("#projects .project")
      assert_equal 1, projects.size, path
      text = projects.first.text
      assert_includes text, org, path
      assert_includes text, period, path
      assert_includes text, "VaivGem-9B", path
      refute_includes text, "BLEU", "#{path}: keep it brief, no metrics"
    end
  end

  def test_mfds_project_team_and_role
    { "index.html" => ["SKI-ML Lab", "Head researcher", "PI", "Howard Lee", "Co-PI", "Jay-Yoon Lee"],
      "ko/index.html" => ["SKI-ML", "head 연구원", "연구책임자", "이형기", "공동연구책임자", "이재윤"] }.each do |path, words|
      project = page(path).at("#project-mfds")
      words.each { |w| assert_includes project.text, w, path }
      hrefs = project.css("a").map { |a| a["href"] }
      assert_includes hrefs, "https://ccadd.snu.ac.kr/", path
      assert_includes hrefs, "https://www.skiml.snu.ac.kr/", path
    end
  end
end

class NewResearchPagesTest < Minitest::Test
  def test_pillar_cards_link_to_their_pages
    { "index.html" => "", "ko/index.html" => "/ko" }.each do |path, pre|
      links = page(path).css("#research .pillar").map { |c| c.at(".pillar__more")&.[]("href") }
      assert_equal ["#{pre}/research/speed/", "#{pre}/research/lsc/", "#{pre}/research/translation-inference/"], links, path
    end
  end

  def test_speed_page_matches_paper
    %w[research/speed/index.html ko/research/speed/index.html].each do |path|
      doc = page(path)
      assert_equal SPEED_TITLE, doc.at(".research__title").text.strip, path
      %w[tldr problem method results citation].each { |id| refute_nil doc.at_css("section##{id}"), "#{path}: ##{id}" }
      refute_nil doc.at("svg.fig-speed[role=img] title"), "#{path}: SPEED figure"
      table = doc.at("table.speed-results")
      refute_nil table, path
      row = ->(k) { table.at(%(tbody tr[data-k="#{k}"])).css("td.num").map { |td| td.text.strip } }
      assert_equal ["51.4", "–", "–", "–"], row.("32"), path
      assert_equal ["51.2", "+33%", "+22%", "25.0%"], row.("24"), path
      assert_equal ["45.4", "+101%", "+55%", "50.0%"], row.("16"), path
      assert_includes doc.at("#citation pre").text, "@article{oh2026shallow", path
    end
  end

  def test_translation_inference_page
    %w[research/translation-inference/index.html ko/research/translation-inference/index.html].each do |path|
      doc = page(path)
      %w[tldr templates loop].each { |id| refute_nil doc.at_css("section##{id}"), "#{path}: ##{id}" }
      text = doc.at("main").text
      ["Retrieval Few-shot", "Segment", "62.3", "58.7", "Length", "Placeholder", "Copy", "QE"].each do |w|
        assert_includes text, w, path
      end
      refute_nil doc.at("svg.fig-loop[role=img] title"), "#{path}: loop figure"
      assert_nil doc.at("#citation"), "#{path}: no paper, no citation block"
      ["corpus", "glossary", "말뭉치", "용어집", "Dictionary", "사전 hint"].each { |w| refute_includes text, w, path }
    end
  end

  def test_mfds_card_leaves_out_data_construction_and_links_to_inference_page
    { "index.html" => "/research/translation-inference/", "ko/index.html" => "/ko/research/translation-inference/" }.each do |path, href|
      card = page(path).at("#project-mfds")
      ["corpus", "glossary", "말뭉치", "용어집"].each { |w| refute_includes card.text, w, path }
      assert_includes card.css("a").map { |a| a["href"] }, href, path
    end
  end
end

# MFDS was team work, so it is described with "we".
class TeamVoiceTest < Minitest::Test
  def test_mfds_work_described_as_team
    en = page("index.html").at(".pillar--prompting").text
    refute_includes en, "I compared"
    assert_includes en, "we compared"
    ko = page("ko/index.html").at(".pillar--prompting").text
    assert_includes ko, "저희 팀은"
  end
end

# The section lists interests, not only finished work.
class InterestsWordingTest < Minitest::Test
  def test_section_and_nav_say_research_interests
    { "index.html" => "Research Interests", "ko/index.html" => "연구 관심 분야" }.each do |path, label|
      doc = page(path)
      assert_equal label, doc.at("#h-research").text.strip, path
      assert_equal label, doc.css(".site-nav a")[1].text.strip, path
    end
  end

  def test_architecture_card_states_interest_not_finished_work
    en = page("index.html").at(".pillar--architecture").text
    refute_includes en, "I look at"
    assert_includes en, "I'm interested in linear and hybrid attention"
    ko = page("ko/index.html").at(".pillar--architecture").text
    refute_includes ko, "접근합니다"
    assert_includes ko, "관심이 있습니다"
  end
end

class ProfilePhotoTest < Minitest::Test
  def test_hero_shows_profile_photo
    { "index.html" => "Jungsuk Oh", "ko/index.html" => "오정석" }.each do |path, name|
      doc = page(path)
      img = doc.at(".hero img.hero__photo")
      refute_nil img, path
      assert_equal "/assets/img/profile.jpg", img["src"], path
      assert_equal name, img["alt"], path
      assert_nil doc.at(".hero__photo--monogram"), "#{path}: monogram replaced"
    end
    assert File.exist?(File.join(SITE, "assets/img/profile.jpg"))
  end
end

# Clicking the email copies the address; mailto: stays as the no-JS fallback.
class CopyEmailMarkupTest < Minitest::Test
  def test_email_links_are_copy_buttons_with_mailto_fallback
    { "index.html" => "Copied!", "ko/index.html" => "복사됨" }.each do |path, copied|
      doc = page(path)
      links = doc.css("a.js-copy-email")
      assert_equal 2, links.size, "#{path}: hero + footer"
      links.each do |a|
        assert_equal "mailto:luke0112@snu.ac.kr", a["href"], path
        assert_equal "luke0112@snu.ac.kr", a["data-email"], path
        assert_equal copied, a["data-copied"], path
        assert_includes a["title"].to_s, "luke0112@snu.ac.kr", path
      end
      refute_nil doc.at("#copy-status[aria-live=polite]"), "#{path}: live region"
    end
  end
end
