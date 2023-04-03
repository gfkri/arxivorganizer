import email
import imaplib
import logging
import pathlib
import re
import shutil
import urllib
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
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from whoosh.query import Regex, Query
from credentials import *
from config import *
import sys

# add parent folder to python path for jinja to find it
sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[1]))

logging.basicConfig(level=logging.INFO)
PaperCollection = recordclass('PaperCollection', 'id title info datetime papers')
Paper = recordclass('Paper', 'id title abstract authors comment published '
                             'hit_terms score arxiv_url pdf_url gs_url supp_url pub_url')


########################################################################################################################
def create_gs_url(title):
  return 'https://scholar.google.com/scholar?q=' + urllib.parse.quote_plus(title)


########################################################################################################################
def generate_website(fp, title, info, papers):
  env = Environment(
    loader=PackageLoader("arxivorganizer"),
    autoescape=select_autoescape()
  )
  template = env.get_template("template.html")
  output = template.render(title=title, info=info, papers=papers)

  with open(fp, 'w') as f:
    f.write(output)


########################################################################################################################
def fetch_arxiv_info(newsletters):
  client = arxiv.Client(
    page_size=1000,
    delay_seconds=3,
    num_retries=5
  )

  paper_id_list = [p for nl in newsletters for p in nl.papers]
  paper_chunks = [paper_id_list[i:i + MAX_ARXIV_REQUESTS] for i in range(0, len(paper_id_list), MAX_ARXIV_REQUESTS)]
  papers = []

  for pchunk in paper_chunks:
    search = arxiv.Search(id_list=pchunk, max_results=len(pchunk), sort_by=arxiv.SortCriterion.SubmittedDate)

    for p in client.results(search):
      authors = ', '.join([a.name for a in p.authors])
      # search_string += tag
      paper = Paper(id=p.get_short_id(), title=p.title, abstract=p.summary.replace('\n', ' '), authors=authors,
                    comment=p.comment, published=p.published, hit_terms=None, score=0.0, arxiv_url=p.entry_id,
                    pdf_url=p.pdf_url, gs_url=create_gs_url(p.title), supp_url=None, pub_url=None)
      papers.append(paper)

  cidx = 0
  for nl in newsletters:
    nl.papers = {p.id: p for p in papers[cidx:cidx+len(nl.papers)]}
    cidx += len(nl.papers)



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
    if msg_index_dp.exists():
      shutil.rmtree(msg_index_dp)

    msg_index_dp.mkdir(parents=True)
    ix = create_in(str(msg_index_dp), schema)

    writer = ix.writer()
    for p in col.papers.values():
      writer.add_document(id=str(p.id), title=p.title, abstract=p.abstract, authors=p.authors,
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
        logging.debug(result['title'], result['authors'], '%.2f' % result.score,
                      '[' + ', '.join(hit_terms) + ']', sep=' | ')
        
    logging.debug('Adding papers, which were not returned by the search ...')
        
    # add papers that were not found in the search
    found_paper_ids = set(map(lambda x:x.id, papers))
    missing_paper_ids = col.papers.keys() - found_paper_ids
    for pid in missing_paper_ids:
      paper = col.papers[pid]
      paper.hit_terms = []
      paper.score = 0.0
      papers.append(paper)
      logging.debug(paper.title, paper.authors, sep=' | ')

    output_fn = output_dp / (col.id + '.html')
    generate_website(output_fn, col.title, col.info, papers)


########################################################################################################################
def fetch_newsletter_from_imap(server_name, username, password, mailfolder, last_n_newsletter, filter_seen=False):
  # create an IMAP4 class with SSL
  imap = imaplib.IMAP4_SSL(server_name)

  # authenticate
  imap.login(username, password)
  _, messages = imap.select(mailfolder)
  # total number of emails
  messages = int(messages[0])

  if IMAP_SERVER_SUPPORTS_SORTING:
    _, sort_order = imap.sort('DATE', 'UTF-8', 'ALL')
    sort_order = sort_order[0].decode('utf-8').split(' ')
  else:
    sort_order = [str(i) for i in range(1, messages+1)]

  newsletters = []
  for i in range(messages - 1, messages - last_n_newsletter - 1, -1):
    _, msg_flags = imap.fetch(sort_order[i], '(FLAGS)')
    seen_flag = np.any([b'FLAGS' in response and b"\\Seen" in response for response in msg_flags])
    if filter_seen and seen_flag:
      continue
            
    _, msg_header = imap.fetch(sort_order[i], '(RFC822)')
    for response in msg_header:    
      if isinstance(response, tuple):
        message = email.message_from_bytes(response[1])
        # decode the email subject
        title, encoding = decode_header(message['Subject'])[0]
        msg_datetime = datetime.strptime(message['Date'], "%a, %d %b %Y %H:%M:%S %z")

        if isinstance(title, bytes):
          title = title.decode(encoding)

        if np.any([kw in title for kw in SUBJECT_SEARCH_STRING]):
          logging.info(f"Analyzing mail with subject '{title}' ...")
          msg_index_dn = msg_datetime.strftime("%Y%m%d%H%M") + '_' + ''.join(title.split(' ')[-2:])

          body = message.get_payload(decode=True).decode()
          ids = re.findall(r"(?<=arXiv:)\d{4}\.\d{5}", body)     

    info = msg_datetime.strftime("%d-%m-%Y %H:%M") + (', %d papers' % len(ids))
    newsletter = PaperCollection(msg_index_dn, title, info, msg_datetime, ids)
    newsletters.append(newsletter)
  
  return newsletters


########################################################################################################################
def main():
  output_dp = pathlib.Path(OUTPUT_DIR)

  newsletters = fetch_newsletter_from_imap(server_name=SERVER_NAME, username=USERNAME, password=PASSWORD,
                                           mailfolder=MAIL_FOLDER, last_n_newsletter=LAST_N_NEWSLETTERS, 
                                           filter_seen=FILTER_SEEN_MESSAGES)
  if IGNORE_ALREADY_CREATED:
    newsletters = [nl for nl in newsletters if not (output_dp / (nl.id + '.html')).exists()]

  fetch_arxiv_info(newsletters)

  # add "overview newsletter" containing all papers, only reasonable if there are more than one newsletter
  if CREATE_OVERVIEW and len(newsletters) > 1:
    papers = {}
    for nl in newsletters:
      papers.update(nl.papers)

    dates = [nl.datetime for nl in newsletters]
    from_date, to_date = min(dates), max(dates)
    title = 'Newsletter Overview [' + from_date.strftime("%Y-%m-%d") + ', ' + to_date.strftime("%Y-%m-%d") + ']'
    info = '%d newsletter, %d papers' % (len(newsletters), len(papers))

    overview_id = 'ov_' + to_date.strftime("%Y%m%d%H%M") + ('_%dnl' % len(newsletters))
    overview_collection = PaperCollection(id=overview_id, title=title, info=info, datetime=to_date, papers=papers)
    newsletters.append(overview_collection)

  sort_and_create(output_dp, newsletters)


########################################################################################################################
if __name__ == '__main__':
  main()
