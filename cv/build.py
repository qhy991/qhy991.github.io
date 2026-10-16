"""Build both public CVs from content.json with embedded Chinese fonts."""
import argparse
import json
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parent
BLACK = colors.HexColor('#141b22')
MUTED = colors.HexColor('#4a5661')
LINK = colors.HexColor('#234f6c')


def build(data, locale):
    text = data['locales'][locale]
    output = ROOT / locale / 'CV.pdf'
    output.parent.mkdir(exist_ok=True)
    page_width, page_height = A4
    c = canvas.Canvas(str(output), pagesize=A4, invariant=1, pageCompression=1)
    c.setTitle(f"{text['name']} CV")
    c.setAuthor('Haiyan Qin')
    c.setSubject(text['tagline'])
    c.setCreator('Haiyan Qin CV renderer')
    zh = locale == 'zh'
    font = 'CVChinese' if zh else 'Helvetica'
    bold = 'CVChineseBold' if zh else 'Helvetica-Bold'
    pdfmetrics.registerFontFamily(font, normal=font, bold=bold, italic=font, boldItalic=bold)
    left, right, y = 36, page_width - 36, page_height - 33
    width = right - left
    body_size = 10.0 if zh else 9.6
    leading = 15.0 if zh else 13.1
    styles = {
        'body': ParagraphStyle('body', fontName=font, fontSize=body_size, leading=leading, textColor=BLACK, wordWrap='CJK' if zh else None, alignment=TA_LEFT),
        'heading': ParagraphStyle('heading', fontName=bold, fontSize=11.8, leading=15.8, textColor=colors.black),
        'project': ParagraphStyle('project', fontName=bold, fontSize=10.0, leading=14, textColor=colors.black),
        'small': ParagraphStyle('small', fontName=font, fontSize=8.2, leading=11.5, textColor=MUTED, wordWrap='CJK' if zh else None),
    }

    def paragraph(value, style='body', available=None, x=None, gap=0):
        nonlocal y
        p = Paragraph(value, styles[style])
        _, height = p.wrap(available or width, page_height)
        if y - height < 30:
            raise ValueError(f'{locale}: content exceeds one-page CV at {value[:65]!r}')
        p.drawOn(c, left if x is None else x, y - height)
        y -= height + gap

    def heading(value):
        nonlocal y
        y -= 8
        paragraph(escape(value), 'heading', gap=4)

    def bullet(value):
        paragraph('&#8226; ' + escape(value), available=width - 3, x=left + 3, gap=1.2)

    def linked(value, url):
        return f'<link href="{escape(url, quote=True)}" color="{LINK.hexval()}">{escape(value)}</link>'

    photo_width, photo_height = 61, 80
    c.drawImage(str(ROOT / 'portrait.jpg'), right - photo_width, y - photo_height,
                width=photo_width, height=photo_height, preserveAspectRatio=True, anchor='n', mask='auto')
    name_font = bold
    c.setFont(name_font, 25)
    c.setFillColor(colors.black)
    c.drawString(left, y - 24, text['name'])
    secondary_font = 'Helvetica' if zh else 'CVChinese'
    c.setFont(secondary_font, 11)
    c.setFillColor(MUTED)
    c.drawString(left + pdfmetrics.stringWidth(text['name'], name_font, 25) + 13, y - 23, text['name_secondary'])
    y -= 36
    paragraph(escape(text['tagline']), 'project', available=width - 78, gap=3)
    paragraph(escape(text['identity']), 'small', available=width - 78, gap=4)
    paragraph(linked(data['email'], 'mailto:' + data['email']), 'small', available=width - 78, gap=1)
    contacts = [linked('qhy991.github.io', data['website']), linked('GitHub', data['github']), linked('Google Scholar', data['scholar'])]
    paragraph(' &nbsp; | &nbsp; '.join(contacts), 'small', available=width - 78, gap=7)
    paragraph(escape(text['summary']), gap=0)
    heading(text['labels']['education'])
    for row in text['education']:
        paragraph(escape(row['title']) + ' &nbsp; <font color="#4a5661">' + escape(row['date']) + '</font>', gap=1)
    heading(text['labels']['achievements'])
    for value in text['achievements']:
        bullet(value)
    heading(text['labels']['projects'])
    for item in text['projects']:
        name = linked(item['name'], item['url']) if item.get('url') else escape(item['name'])
        paragraph(name + ' &nbsp; <font name="' + font + '" size="8.3" color="#4a5661">' + escape(item['role']) + '</font>', 'project', gap=2)
        for value in item['bullets']:
            bullet(value)
        y -= 3
    heading(text['labels']['publications'])
    for item in text['publications']:
        paragraph('<b>' + linked(item['name'], item['url']) + '</b> &nbsp; ' + escape(item['venue']), gap=1)
        paragraph(escape(item['title']), 'small', gap=4)
    heading(text['labels']['skills'])
    for value in text['skills']:
        paragraph(escape(value), 'small', gap=1)
    c.setFont(font, 7)
    c.setFillColor(MUTED)
    c.drawRightString(right, 17, f"{text['labels']['updated']} {data['updated']}")
    c.showPage()
    c.save()
    print(f'{output.relative_to(ROOT)}: one A4 page, body ends at {y:.1f} pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-zh', type=Path, default=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
    parser.add_argument('--font-zh-bold', type=Path, default=Path('/System/Library/Fonts/STHeiti Medium.ttc'))
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont('CVChinese', str(args.font_zh)))
    pdfmetrics.registerFont(TTFont('CVChineseBold', str(args.font_zh_bold)))
    data = json.loads((ROOT / 'content.json').read_text())
    for locale in data['locales']:
        build(data, locale)


if __name__ == '__main__':
    main()
