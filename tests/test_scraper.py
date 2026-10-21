from scraper import Article, build_markdown, html_to_markdown, save_articles, slugify


def test_saved_files_use_lf_on_every_os(tmp_path):
    # CRLF on Windows vs LF in Docker would make every hash differ and re-upload everything.
    save_articles([Article(1, "T", "u", "d", "t.md", "a\nb\n")], tmp_path)
    assert (tmp_path / "t.md").read_bytes() == b"a\nb\n"


def test_headings_are_atx_without_bold():
    md = html_to_markdown("<h2><strong>Here are the steps:</strong></h2>")
    assert md.strip() == "## Here are the steps:"


def test_body_h1_is_demoted_below_title():
    md = html_to_markdown("<h1>Setup</h1><h2>Step</h2>")
    assert "## Setup" in md and "### Step" in md
    assert not any(line.startswith("# ") for line in md.splitlines())


def test_code_block_preserved_verbatim():
    md = html_to_markdown(r"<pre>cd c:\temp\n_a_ *b*</pre>")
    assert "```\ncd c:\\temp\\n_a_ *b*\n```" in md


def test_literal_markdown_chars_not_escaped():
    md = html_to_markdown("<p>***ALL*** and VIDEO_ID and a | b</p>")
    assert "\\" not in md
    assert "***ALL***" in md and "VIDEO_ID" in md


def test_in_page_anchor_links_resolve():
    md = html_to_markdown('<a href="#Step1">Go</a><a name="Step1"></a><h2>Step 1</h2>')
    assert "[Go](#Step1)" in md
    assert '<a id="Step1"></a>' in md


def test_site_relative_links_made_absolute_others_untouched():
    md = html_to_markdown(
        '<a href="/hc/en-us/articles/1">A</a> <a href="mailto:x@y.com">M</a> '
        '<a href="https://ex.com/p">E</a>'
    )
    assert "(https://support.optisigns.com/hc/en-us/articles/1)" in md
    assert "(mailto:x@y.com)" in md
    assert "(https://ex.com/p)" in md


def test_image_gets_own_paragraph():
    md = html_to_markdown('<p><img src="https://x/i.png" alt="pic">Once logged in</p>')
    assert "![pic](https://x/i.png)\n\nOnce logged in" in md


def test_table_has_header_row():
    md = html_to_markdown("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>")
    assert "| A | B |\n| --- | --- |\n| 1 | 2 |" in md


def test_iframe_becomes_link():
    md = html_to_markdown('<iframe src="//www.youtube.com/embed/abc"></iframe>')
    assert "[YouTube video](https://www.youtube.com/embed/abc)" in md


def test_nav_junk_removed():
    md = html_to_markdown("<script>x()</script><style>p{}</style><p>Keep</p><button>Nope</button>")
    assert md.strip() == "Keep"


def test_build_markdown_has_title_and_citable_url():
    raw = {"title": " T ", "html_url": "https://u", "updated_at": "2026-01-01", "body": "<p>b</p>"}
    md = build_markdown(raw)
    assert md.startswith("# T\n\nArticle URL: https://u\n")


def test_slugify():
    assert slugify("How to Add a YouTube Video?!") == "how-to-add-a-youtube-video"
