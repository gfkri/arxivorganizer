import csv
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
import tqdm as tqdm
import whoosh.query
from jinja2 import Environment, PackageLoader, select_autoescape
from recordclass import recordclass
from whoosh import scoring
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.query import Regex, Query
from credentials import *
from config import *
import sys
from fuzzywuzzy import fuzz


# add parent folder to python path for jinja to find it
sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[1]))

logging.basicConfig(level=logging.INFO)
PaperCollection = recordclass('PaperCollection', 'id title datetime papers')
Paper = recordclass('Paper', 'id title abstract authors comment hit_terms score arxiv_url pdf_url')


########################################################################################################################
def generate_website(fp, title, papers):
  env = Environment(
    loader=PackageLoader("arxivorganizer"),
    autoescape=select_autoescape()
  )
  template = env.get_template("template.html")
  output = template.render(title=title, papers=papers)

  with open(fp, 'w') as f:
    f.write(output)


########################################################################################################################
def main():
  output_dp = pathlib.Path(OUTPUT_DIR)

  newsletters = fetch_newsletter_from_imap(server_name=SERVER_NAME, username=USERNAME, password=PASSWORD,
                                           mailfolder=MAIL_FOLDER, last_n_newsletter=LAST_N_NEWSLETTERS)

  # add "overview newsletter" containing all papers, only reasonable if there are more than one newsletter
  if CREATE_OVERVIEW and len(newsletters) > 1:
    papers = {}
    for nl in newsletters:
      papers.update(nl.papers)

    dates = [nl.datetime for nl in newsletters]
    from_date, to_date = min(dates), max(dates)
    title = 'Newsletter Overview [' + from_date.strftime("%Y-%m-%d") + ', ' + to_date.strftime("%Y-%m-%d") + '] ' + \
            '(%d newsletter, %d papers)' % (len(newsletters), len(papers))

    overview_id = 'ov_' + to_date.strftime("%Y%m%d%H%M") + ('_%dnl' % len(newsletters))
    overview_collection = PaperCollection(id=overview_id, title=title, datetime=to_date, papers=papers)
    newsletters.append(overview_collection)

  sort_and_create(output_dp, newsletters)


########################################################################################################################
def sort_and_create(output_dp, collections, index_dp=pathlib.Path(INDEX_DIR)):
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

  # search for keywords in each newsletter and create a website
  for col in collections:
    msg_index_dp = index_dp / col.id
    if not msg_index_dp.exists():
      msg_index_dp.mkdir(parents=True)
    ix = open_dir(str(msg_index_dp))
    writer = ix.writer()
    for p in col.papers.values():
      print(p.id)
      writer.add_document(id=p.id, title=p.title, abstract=p.abstract, authors=p.authors,
                          comment=p.comment, arxiv_url=p.arxiv_url, pdf_url=p.pdf_url)
    writer.commit()

    papers = []
    with ix.searcher(weighting=mw) as searcher:
      results = searcher.search(q, limit=None, terms=True)
      for result in results:
        hit_terms = sorted(list({t[1].decode('utf-8') for t in result.matched_terms()}))
        paper = col.papers[result['id']]
        paper.hit_terms = hit_terms
        paper.score = result.score
        papers.append(paper)
        logging.debug(result['title'], result['arxiv_url'], result['authors'], '%.2f' % result.score,
                      '[' + ', '.join(hit_terms) + ']', sep=' | ')

    output_fn = output_dp / (col.id + '.html')
    generate_website(output_fn, col.title, papers)


########################################################################################################################
def fetch_newsletter_from_imap(server_name, username, password, mailfolder, last_n_newsletter):
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
    newsletter = PaperCollection(msg_index_dn, title, msg_datetime, papers)
    newsletters.append(newsletter)
  return newsletters


########################################################################################################################
def fetch_newsletter_from_csv(csv_file, delimiter=',', quotechar='"', encoding=None, fuzzy_th=90,
                              filter_fn=lambda x: True):
  # fetches archive information from a csv, which needs to have field Title
  papers = {}
  with open(csv_file, newline='', encoding=encoding) as csvfile:
    reader = csv.DictReader(csvfile, delimiter=delimiter, quotechar=quotechar)
    for row in tqdm.tqdm(reader):
      if filter_fn(row):
        search = arxiv.Search(query='ti:%s' % row['Title'].replace(':', ''), max_results=10)
        results = list(search.results())

        for r in results:
          if fuzz.ratio(row['Title'].lower(), r.title.lower()) > fuzzy_th:
            authors = ', '.join([a.name for a in r.authors])
            # search_string += tag
            paper = Paper(id=r.get_short_id(), title=r.title, abstract=r.summary.replace('\n', ' '), authors=authors,
                          comment=r.comment, hit_terms=None, score=0.0, arxiv_url=r.entry_id, pdf_url=r.pdf_url)
            papers[r.get_short_id()] = paper
            break
  title = subject + ' (' + msg_datetime.strftime("%d-%m-%Y %H:%M") + (', %d papers)' % len(papers))
  newsletter = PaperCollection(msg_index_dn, title, msg_datetime, papers)
  newsletters.append(newsletter)
  return papers


########################################################################################################################
if __name__ == '__main__':
  #csv_file = '/home/gfkri/Downloads/paper list per session.csv'
  #papers = fetch_newsletter_from_csv(csv_file)

  main()
