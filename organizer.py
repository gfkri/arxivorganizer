import email
import imaplib
import logging
import os
import pathlib
import re
from datetime import datetime
from email.header import decode_header
from functools import reduce

import arxiv
import numpy as np
import whoosh.query
from jinja2 import Environment, PackageLoader, select_autoescape
from recordclass import recordclass
from whoosh import scoring
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.query import Every, Regex, Query
from credentials import *
from config import *
import sys
import itertools
import hashlib, base64


# add parent folder to python path for jinja to find it
sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[1]))

logging.basicConfig(level=logging.INFO)
Newsletter = recordclass('Newsletter', 'id title datetime papers')
Paper = recordclass('Paper', 'id title abstract authors comment hit_terms score arxiv_url pdf_url')


########################################################################################################################
def create_website(fp, title, papers):
  env = Environment(
    loader=PackageLoader("arxivorganizer"),
    autoescape=select_autoescape()
  )
  template = env.get_template("template.html")
  output = template.render(newsletter_title=title, papers=papers)

  with open(fp, 'w') as f:
    f.write(output)


########################################################################################################################
def main():
  index_dp = pathlib.Path(INDEX_DIR)
  output_dp = pathlib.Path(OUTPUT_DIR)
  search_keyword_string = ' '.join(SEARCH_KEYWORDS)

  # search engine initialization
  stem_ana = StemmingAnalyzer()
  schema = Schema(id=ID(stored=True, sortable=True),
                  title=TEXT(analyzer=stem_ana, stored=True, sortable=True, phrase=True),
                  abstract=TEXT(analyzer=stem_ana, stored=True, sortable=True, phrase=True),
                  authors=TEXT(stored=True, sortable=True, phrase=True),
                  arxiv_url=ID(stored=True, sortable=True),
                  pdf_url=ID(stored=True, sortable=True),
                  comment=TEXT(analyzer=stem_ana, stored=True, sortable=True))

  q_search_title = reduce(Query.__or__, [QueryParser('title', schema).parse(kw) for kw in SEARCH_KEYWORDS])
  q_search_keywords = reduce(Query.__or__, [QueryParser('abstract', schema).parse(kw) for kw in SEARCH_KEYWORDS])
  q_search_authors = reduce(Query.__or__, [QueryParser('authors', schema).parse(kw).with_boost(0.75)
                                           for kw in SEARCH_AUTHORS])
  q_search_conferences = reduce(Regex.__or__, [Regex('comment', kw, boost=0.5) for kw in SEARCH_CONFERENCES])
  q = q_search_title | q_search_keywords | q_search_authors | q_search_conferences | \
      whoosh.query.Every('arxiv_url', boost=0.001)
  mw = scoring.MultiWeighting(scoring.BM25F())

  newsletters = fetch_newsletter(server_name=SERVER_NAME, username=USERNAME, password=PASSWORD,
                                 mailfolder=MAIL_FOLDER, last_n_newsletter=LAST_N_NEWSLETTERS)

  # add "overview newsletter" containing all papers
  if CREATE_OVERVIEW:
    papers = {}
    for nl in newsletters:
      papers.update(nl.papers)

    dates = [nl.datetime for nl in newsletters]
    from_date, to_date = min(dates), max(dates)
    title = 'Newsletter Overview [' + from_date.strftime("%Y-%m-%d") + ', ' + to_date.strftime("%Y-%m-%d") + '] ' + \
            '(%d newsletter, %d papers)' % (len(newsletters), len(papers))

    newsletter_id = 'ov_' + to_date.strftime("%Y%m%d%H%M") + ('_%dnl' % len(newsletters))
    overview_newsletter = Newsletter(id=newsletter_id, title=title, datetime=to_date, papers=papers)
    newsletters.append(overview_newsletter)

  # search for keywords in each newsletter and create a website
  for nl in newsletters:
    msg_index_dp = index_dp / nl.id
    if not msg_index_dp.exists():
      os.mkdir(msg_index_dp)
    ix = create_in(str(msg_index_dp), schema)
    ix = open_dir(str(msg_index_dp))
    writer = ix.writer()
    for p in nl.papers.values():
      writer.add_document(id=p.id, title=p.title, abstract=p.abstract, authors=p.authors,
                          comment=p.comment, arxiv_url=p.arxiv_url, pdf_url=p.pdf_url)
    writer.commit()

    papers = []
    with ix.searcher(weighting=mw) as searcher:
      results = searcher.search(q, limit=None, terms=True)
      for result in results:
        hit_terms = sorted(list({t[1].decode('utf-8') for t in result.matched_terms()}))
        paper = nl.papers[result['id']]
        paper.hit_terms = hit_terms
        paper.score = result.score
        papers.append(paper)
        logging.debug(result['title'], result['arxiv_url'], result['authors'], '%.2f' % result.score,
                      '[' + ', '.join(hit_terms) + ']', sep=' | ')


    output_fn = output_dp / (nl.id + '.html')
    create_website(output_fn, nl.title, papers)


########################################################################################################################
def fetch_newsletter(server_name, username, password, mailfolder, last_n_newsletter):
  # create an IMAP4 class with SSL
  imap = imaplib.IMAP4_SSL(server_name)
  # authenticate
  imap.login(username, password)
  status, messages = imap.select(mailfolder)
  # total number of emails
  messages = int(messages[0])
  newsletters = []
  for i in range(messages, messages - last_n_newsletter, -1):
    res, msg = imap.fetch(str(i), '(RFC822)')
    papers = {}

    for response in msg:
      if isinstance(response, tuple):
        msg = email.message_from_bytes(response[1])
        # decode the email subject
        subject, encoding = decode_header(msg['Subject'])[0]
        msg_datetime = datetime.strptime(msg['Date'], "%a, %d %b %Y %H:%M:%S %z")

        if isinstance(subject, bytes):
          subject = subject.decode(encoding)

        if np.any([kw in subject for kw in SUBJECT_SEARCH_STRING]):
          logging.info(f"Analyzing mail with subject '{subject}' ...")
          msg_index_dn = msg_datetime.strftime("%Y%m%d%H%M") + '_' + ''.join(subject.split(' ')[-2:])

          body = msg.get_payload(decode=True).decode()
          ids = re.findall(r"(?<=arXiv:)\d{4}\.\d{5}", body)
          search = arxiv.Search(id_list=ids, max_results=len(ids), sort_by=arxiv.SortCriterion.SubmittedDate)
          results = list(search.results())

          for r in results:
            authors = ', '.join([a.name for a in r.authors])
            # search_string += tag
            paper = Paper(id=r.get_short_id(), title=r.title, abstract=r.summary.replace('\n', ' '), authors=authors,
                          comment=r.comment, hit_terms=None, score=0.0, arxiv_url=r.entry_id, pdf_url=r.pdf_url)
            papers[r.get_short_id()] = paper

    title = subject + ' (' + msg_datetime.strftime("%d-%m-%Y %H:%M") + (', %d papers)' % len(papers))
    newsletter = Newsletter(msg_index_dn, title, msg_datetime, papers)
    newsletters.append(newsletter)
  return newsletters


########################################################################################################################
if __name__ == '__main__':
  main()
