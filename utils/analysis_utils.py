import csv
import logging
import pathlib
import pickle
import sys
from datetime import datetime
from arxiv import arxiv

from fuzzywuzzy import fuzz
from tqdm import tqdm

import logging

def _init_logger():
  logger = logging.getLogger(__file__)
  logger.setLevel(logging.INFO)
  return logger
  
_logger = _init_logger()

from config import OUTPUT_DIR, INDEX_DIR, PAPER_CACHE_DIR
from organizer import PaperCollection, sort_and_create, Paper, create_gs_url

import requests
from bs4 import BeautifulSoup

# add parent folder to python path for jinja to find it
sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[2]))

logging.basicConfig(level=logging.INFO)
OPEN_ACCESS_URL = "https://openaccess.thecvf.com"
ECVA_PAPERS_URL = "https://www.ecva.net"
NEURIPS_PAPERS_URL = "https://proceedings.neurips.cc"


########################################################################################################################
def fetch_papers_from_csv(csv_file, delimiter=',', quotechar='"', encoding=None, fuzzy_th=90,
                              filter_fn=lambda x: True):
  # fetches archive information from a csv, which needs to have field Title
  papers = {}
  not_found_papers = {}
  with open(csv_file, newline='', encoding=encoding) as csvfile:
    reader = csv.DictReader(csvfile, delimiter=delimiter, quotechar=quotechar)
    for row in tqdm(reader):
      if filter_fn(row):
        not_found_papers[row['Paper ID']] = Paper(paper_id=row['Paper ID'], title=row['Title'], 
                                                  authors=row['Authors'].replace(';', ','), comment='ICCV 2021',
                                                  gs_url=create_gs_url(row['Title']))

        logging.info(f"Searching for paper with title '{row['Title']}' ...")
        search = arxiv.Search(query='ti:%s' % row['Title'].replace(':', ''), max_results=10)
        results = list(search.results())

        for p in results:
          if fuzz.ratio(row['Title'].lower(), p.title.lower()) > fuzzy_th:
            del not_found_papers[row['Paper ID']]
            logging.info(f"Found paper with title '{row['Title']}'.")
            authors = ', '.join([a.name for a in p.authors])
            paper = Paper(paper_id=p.get_short_id(), title=p.title, abstract=p.summary.replace('\n', ' '), authors=authors,
                          comment=p.comment, arxiv_url=p.entry_id, pdf_url=p.pdf_url, gs_url=create_gs_url(p.title), published=p.published)
            papers[p.get_short_id()] = paper
            break
          
  return {**papers, **not_found_papers}


########################################################################################################################
def fetch_papers_from_text(text_file, encoding=None, fuzzy_th=90):
  # fetches archive information from a csv, which needs to have field Title
  papers = {}
  not_found_papers = {}
  with open(text_file, newline='', encoding=encoding) as txtfile:
    for idx, title in tqdm(enumerate(txtfile)):
      title = title.strip()
      paper_id = '%05d' % idx
      not_found_papers[paper_id] = Paper(paper_id=paper_id, title=title, gs_url=create_gs_url(title))

      logging.info(f"Searching for paper with title '{title}' ...")
      search = arxiv.Search(query='ti:%s' % title.replace(':', ''), max_results=10)
      results = list(search.results())

      for p in results:
        if fuzz.ratio(title.lower(), p.title.lower()) > fuzzy_th:
          del not_found_papers[paper_id]
          logging.info(f"Found paper with title '{title}'.")
          authors = ', '.join([a.name for a in p.authors])
          paper = Paper(paper_id=p.get_short_id(), title=p.title, abstract=p.summary.replace('\n', ' '), authors=authors,
                        comment=p.comment, arxiv_url=p.entry_id, pdf_url=p.pdf_url, gs_url=create_gs_url(p.title), 
                        published=p.published)
          papers[p.get_short_id()] = paper
          break
  return {**papers, **not_found_papers}


########################################################################################################################
def parse_openaccess(conference, conference_appendices):
  papers = {}
  for appendix in conference_appendices:
    page = requests.get(OPEN_ACCESS_URL + appendix)

    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("dt", {"class": "ptitle"})
    results = [r.find('a') for r in results]
    results = {r.text: r.attrs['href'] for r in results}

    for idx, (title, url) in tqdm(enumerate(results.items()), total=len(results)):
      paper_id = '%05d' % idx
      pub_url = OPEN_ACCESS_URL + url
      page = requests.get(pub_url)
      if page.status_code != 200:
        _logger.warning(f"Could not fetch page '{pub_url}'.")
        continue
      
      soup = BeautifulSoup(page.content, "html.parser")
      content = soup.find("div", {"id": "content"}).find('dl', recursive=False)
      abstract = content.find(id='abstract').text.strip()
      authors = content.find(id='authors').find('i').text

      links = content.find('dd', recursive=False)
      links = {r.text: r.attrs['href'] for r in links.find_all('a', recursive=False) if 'href' in r.attrs}
      pdf_url = (OPEN_ACCESS_URL + links['pdf']) if 'pdf' in links else None
      supp_url = (OPEN_ACCESS_URL + links['supp']) if 'supp' in links else None
      arxiv_url = links['arXiv'] if 'arXiv' in links else None

      paper = Paper(paper_id=paper_id, title=title, abstract=abstract, authors=authors,
                    comment=conference, score=0.0, arxiv_url=arxiv_url, pdf_url=pdf_url,
                    gs_url=create_gs_url(title), supp_url=supp_url, pub_url=pub_url)
      papers[paper_id] = paper
  return papers


########################################################################################################################
def parse_neurips(year):
  papers = {}
  page = requests.get(f'{NEURIPS_PAPERS_URL}/paper_files/paper/{year}')
  conference = f'NeurIPS {year}'

  soup = BeautifulSoup(page.content, "html.parser")
  results = soup.find_all("a", {"title": "paper title"})
  results = {r.text: r.attrs['href'] for r in results}

  for idx, (title, url) in tqdm(enumerate(results.items()), total=len(results)):
    paper_id = '%05d' % idx
    pub_url = NEURIPS_PAPERS_URL + url
    page = requests.get(pub_url)
    if page.status_code != 200:
      _logger.warning(f"Could not fetch page '{pub_url}'.")
      continue
    
    soup = BeautifulSoup(page.content, "html.parser")
    content = soup.find("div", {"class": "col"})
    title = content.find('h4').text
    authors = content.find('i', recursive=True).text
    links = content.find('div').find_all('a')
    links = {r.text: r.attrs['href'] for r in links if 'href' in r.attrs}
    
    abstract_tag = soup.find('h4', text='Abstract')
    abstract = abstract_tag.find_next('p').text.strip() if abstract_tag else ''
    
    pdf_url = f"{NEURIPS_PAPERS_URL}{links['Paper']}" if 'Paper' in links else None
    supp_url = f"{NEURIPS_PAPERS_URL}{links['Supplemental']}" if 'Supplemental' in links else None
    reviews_url = links['Reviews And Public Comment'] if 'Reviews And Public Comment' in links else None

    paper = Paper(paper_id=paper_id, title=title, abstract=abstract, authors=authors,
                  comment=conference, score=0.0, reviews_url=reviews_url, pdf_url=pdf_url,
                  gs_url=create_gs_url(title), supp_url=supp_url, pub_url=pub_url)
    papers[paper_id] = paper
  return papers


########################################################################################################################
def fetch_papers_from_title(papers, fuzzy_th=90, max_requests=1):
  client = arxiv.Client(
    page_size=1000,
    delay_seconds=3,
    num_retries=5
  )
  not_found_papers = {paper.paper_id: paper for paper in papers}
  paper_chunks = [papers[i:i + max_requests] for i in range(0, len(papers), max_requests)]
  for chunk in tqdm(paper_chunks):
    titles = [paper.title for paper in chunk]
    search_query = ' OR '.join([f'(ti:{title})' for title in titles])
    search = arxiv.Search(query=search_query, max_results=max_requests*10)
    for p in client.results(search):
      for paper in chunk:
        if fuzz.ratio(paper.title.lower(), p.title.lower()) > fuzzy_th:
          logging.info(f"Found paper with title '{paper.title}'.")

          paper_id = paper.paper_id
          authors = ', '.join([a.name for a in p.authors])
          paper = Paper(paper_id=paper_id, title=p.title, abstract=p.summary.replace('\n', ' '), authors=authors,
                        comment=p.comment, arxiv_url=p.entry_id, pdf_url=p.pdf_url,
                        gs_url=create_gs_url(p.title), published=p.published)
          del not_found_papers[paper_id]
          papers[paper_id] = paper
          pickle.dump(papers, open('papers.pkl', 'wb'))
          test = pickle.load(open('papers.pkl', 'rb'))
          break
    print('')

  return {**papers, **not_found_papers}


########################################################################################################################
def parse_cvpr(conference, url, fuzzy_th=90, max_requests=50):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, "html.parser")
  title_list = soup.find_all("tr")
  title_author_list = [(r.find('strong'), r.find('i')) for r in title_list]
  title_author_list = [(ti_el.text, au_el.text.replace(' ·', ',')) for ti_el, au_el in title_author_list 
                       if ti_el is not None]
  papers = [Paper(idx, title=ti, authors=au) for idx, (ti, au) in enumerate(title_author_list)]
  fetched_papers = fetch_papers_from_title(papers)
  
  return fetched_papers


########################################################################################################################
def parse_ecva(conference):
  page = requests.get(ECVA_PAPERS_URL + '/papers.php')
  soup = BeautifulSoup(page.content, "html.parser")
  results = soup.find_all("div", {"id": "content"})

  papers = {}
  for conf in results:
    # check if conference, e.g. eccv_2020 somewhere in the text
    if conference.lower().replace(' ', '_') in conf.find('a').attrs['href']:
      paper_data = list(conf.find('dl').find_all(['dt']))
      for idx, element in tqdm(enumerate(paper_data), total=len(paper_data)):
        paper_id = '%05d' % idx
        authors = element.nextSibling.text.strip()
        links = element.nextSibling.nextSibling.nextSibling      
        element = element.find('a') 
        title = element.text.strip() 
        links = {r.text: r.attrs['href'] for r in links.find_all('a')}
        page_url = f"{ECVA_PAPERS_URL}/{element.attrs['href']}"
        pub_url = links['DOI'] if 'DOI' in links else ECVA_PAPERS_URL + element.attrs['href']
        pdf_url = f"{ECVA_PAPERS_URL}/{links['pdf']}" if 'pdf' in links else None
        supp_url = f"{ECVA_PAPERS_URL}/{links['supplementary material']}" if 'supplementary material' in links else None

        page = requests.get(page_url)
        element_soup = BeautifulSoup(page.content, "html.parser")
        abstract = element_soup.find('div', {"id": "abstract"}).text.strip()
        
        # remove staring and trailing quotes if present
        if abstract.startswith('"'): 
          abstract = abstract[1:]
        if abstract.endswith('"'):
          abstract = abstract[:-1]

        paper = Paper(paper_id=paper_id, title=title, abstract=abstract, authors=authors, comment=conference, 
                      pdf_url=pdf_url, gs_url=create_gs_url(title), supp_url=supp_url, pub_url=pub_url)
        papers[paper_id] = paper
  return papers


########################################################################################################################
def ecva_analysis(conference='ECCV 2020'):
  output_dp = pathlib.Path('.') / OUTPUT_DIR
  index_dp = pathlib.Path('.') / INDEX_DIR
  cache_dp = pathlib.Path('.') / PAPER_CACHE_DIR
  cache_dp.mkdir(parents=True, exist_ok=True)
  collection_id = conference.replace(' ', '_').lower()
  cache_fn = cache_dp / f'{collection_id}.json'
  
  if cache_fn.exists():
    _logger.info(f"Loading papers from cache '{cache_fn}' ...")
    collection = PaperCollection.load_from_file(cache_fn)
  else:
    _logger.info(f"Fetching papers for conference '{conference}' from '{ECVA_PAPERS_URL}' ...")
    papers = parse_ecva(conference)
    title = '%s' % (conference)
    info = '%d papers' % len(papers)
    collection = PaperCollection(collection_id, title, info, datetime.now(), papers)
    collection.save_to_file(cache_fn)
    
  sort_and_create(output_dp, [collection], index_dp)

# TODO fuse the functions above and below
########################################################################################################################
def oa_analysis(conference, conference_appendices=None):
  if conference_appendices is None:
    conference_appendices = ['/%s?day=all' % conference.replace(' ', '')]
    
  output_dp = pathlib.Path('.') / OUTPUT_DIR
  index_dp = pathlib.Path('.') / INDEX_DIR
  cache_dp = pathlib.Path('.') / PAPER_CACHE_DIR
  cache_dp.mkdir(parents=True, exist_ok=True)
  collection_id = conference.replace(' ', '_').lower()
  cache_fn = cache_dp / f'{collection_id}.json'
  
  if cache_fn.exists():
    _logger.info(f"Loading papers from cache '{cache_fn}' ...")
    collection = PaperCollection.load_from_file(cache_fn)
  else:
    _logger.info(f"Fetching papers for conference '{conference}' from '{OPEN_ACCESS_URL}' ...")
    papers = parse_openaccess(conference, conference_appendices)  
    title = '%s' % (conference)
    info = '%d papers' % len(papers)
    collection = PaperCollection(collection_id, title, info, datetime.now(), papers)
    collection.save_to_file(cache_fn)
  
  sort_and_create(output_dp, [collection], index_dp)
  
########################################################################################################################
def neurips_analysis(year):   
  output_dp = pathlib.Path('.') / OUTPUT_DIR
  index_dp = pathlib.Path('.') / INDEX_DIR
  cache_dp = pathlib.Path('.') / PAPER_CACHE_DIR
  cache_dp.mkdir(parents=True, exist_ok=True)
  collection_id = 'neurips_%s' % year
  cache_fn = cache_dp / f'{collection_id}.json'
  
  if cache_fn.exists():
    _logger.info(f"Loading papers from cache '{cache_fn}' ...")
    collection = PaperCollection.load_from_file(cache_fn)
  else:
    _logger.info(f"Fetching papers for NeurIPS '{year}' from '{NEURIPS_PAPERS_URL}' ...")
    title = f'NeurIPS {year}'
    papers = parse_neurips(year)
    info = f'{len(papers)} papers'
    collection = PaperCollection(collection_id, title, info, datetime.now(), papers)
    collection.save_to_file(cache_fn)
  
  sort_and_create(output_dp, [collection], index_dp)

########################################################################################################################
def cvpr_analysis(conference, url):
  output_dp = pathlib.Path('.') / OUTPUT_DIR
  index_dp = pathlib.Path('.') / INDEX_DIR

  papers = parse_cvpr(conference, url)
  title = '%s' % (title)
  info = '%d papers' % len(papers)
  newsletters = [PaperCollection(conference.replace(' ', '_').lower(), title, info, datetime.now(), papers)]
  sort_and_create(output_dp, newsletters, index_dp)


########################################################################################################################
def iccv_csv_analysis():
  output_dp = pathlib.Path('.') / OUTPUT_DIR
  index_dp = pathlib.Path('.') / INDEX_DIR
  csv_file = '/home/gfkri/Downloads/paper list per session.csv'

  filter_fn = lambda x: True
  # filter_fn = lambda x: x['Session #'] == 'Session 10'

  papers = fetch_papers_from_csv(csv_file, filter_fn=filter_fn)
  title = 'ICCV 2021'
  info = '%d papers' % len(papers)
  newsletters = [PaperCollection('iccv_2021', title, info, datetime.now(), papers)]
  sort_and_create(output_dp, newsletters, index_dp)


########################################################################################################################
def pc_github_analysis():
  output_dp = pathlib.Path('.') / OUTPUT_DIR
  index_dp = pathlib.Path('.') / INDEX_DIR
  csv_file = 'data/pc_papers_2021.md'

  papers = fetch_papers_from_text(csv_file)
  title = 'Point Cloud Github Repo'
  info = '%d papers' % len(papers)
  newsletters = [PaperCollection('pc_github_2021', title, info, datetime.now(), papers)]
  sort_and_create(output_dp, newsletters, index_dp)


########################################################################################################################
def eccv_csv_analysis():
  output_dp = pathlib.Path('.') / OUTPUT_DIR
  index_dp = pathlib.Path('.') / INDEX_DIR
  csv_file = 'data/eccv_2022.csv'

  filter_fn = lambda x: True
  # filter_fn = lambda x: x['Session #'] == 'Session 10'

  papers = fetch_papers_from_csv(csv_file, filter_fn=filter_fn)
  title = 'ECCV 2022'
  info = '%d papers' % len(papers)
  newsletters = [PaperCollection('eccv_2022', title, info, datetime.now(), papers)]
  sort_and_create(output_dp, newsletters, index_dp)


########################################################################################################################
if __name__ == '__main__':
  ecva_analysis('ECCV 2022')
  # oa_analysis('CVPR 2021')
  # oa_analysis('ICCV 2021')
  # neurips_analysis('2023')
  # oa_analysis('CVPR 2020', [r'CVPR2020?day=2020-06-16', r'CVPR2020?day=2020-06-17', r'CVPR2020?day=2020-06-18'])  

