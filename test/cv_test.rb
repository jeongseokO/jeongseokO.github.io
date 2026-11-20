# Checks the compiled CV (cv/JungsukOh_CV.pdf). Build it first with script/build-cv.
require "minitest/autorun"
require "open3"

PDF = File.expand_path("../cv/JungsukOh_CV.pdf", __dir__)

def pdf_run(*cmd)
  out, status = Open3.capture2(*cmd)
  raise "#{cmd.first} failed" unless status.success?
  out.force_encoding("UTF-8")
end

class CvTest < Minitest::Test
  def text = @text ||= pdf_run("pdftotext", "-enc", "UTF-8", "-layout", PDF, "-")
  def info = @info ||= pdf_run("pdfinfo", PDF)

  def test_pdf_exists
    assert File.exist?(PDF), "build the CV with script/build-cv"
  end

  def test_at_most_two_pages
    assert_operator info[/^Pages:\s+(\d+)/, 1].to_i, :<=, 2
  end

  def test_required_content
    ["Jungsuk Oh", "Ph.D. in Data Science", "Sep 2026", "AI Researcher", "R&DB Foundation",
     "Head Researcher", "Howard Lee", "Latent Self-Consistency", "AAAI", "10.1609/aaai.v40i38.40536",
     "2605.06105", "Shallow Prefill, Deep Decoding", "ICLR 2026", "ICML 2026", "AAAI 2026", "2027",
     "vLLM", "TRL", "Slurm", "Teaching Assistant", "Samsung AI Expert Program"].each { |s| assert_includes text, s }
  end

  def test_removed_content
    ["010-", "GPA", "eGlobal", "LatentCOMP", "Research Loop", "Coursework", "LLM Training Program"].each { |s| refute_includes text, s }
  end

  def test_english_only
    assert_nil text[/\p{Hangul}/], "the CV font has no Hangul glyphs"
  end

  def test_links
    urls = pdf_run("pdfinfo", "-url", PDF)
    ["https://doi.org/10.1609/aaai.v40i38.40536", "https://arxiv.org/abs/2605.06105",
     "https://github.com/jeongseokO/LatentSC", "https://jeongseoko.github.io/research/lsc/",
     "https://jeongseoko.github.io/research/speed/"].each { |u| assert_includes urls, u }
  end

  # Inter's contextual alternates swap punctuation next to capitals for glyphs that
  # copy/paste and applicant tracking systems cannot map back to text.
  def test_punctuation_survives_text_extraction
    ["(SKI-ML Lab)", "(MFDS)", "PI: Prof. Howard Lee", "Co-PI: Prof. Jay-Yoon Lee", "[P1]"].each do |s|
      assert_includes text, s
    end
  end

  def test_mfds_described_once
    refute_includes text, "side of the MFDS translation project"
  end

  def test_metadata
    assert_match(/^Title:.*Jungsuk Oh.*CV/, info)
    assert_match(/^Author:\s+Jungsuk Oh$/, info)
  end
end
