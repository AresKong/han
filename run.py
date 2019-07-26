from preprocess import build_dbank, build_blog
from models import han, hanage, hanblog, gru

def build_datasets():
    build_dbank.build_dbank_json()
    build_blog.build_blog_json()

def evaluations():
    han.evaluate_han(batchsize=16)
    hanage.evaluate_hanage(batchsize=16)
    hanblog.evaluate_hanblog(batchsize=16)
    gru.evaluate_gru(batchsize=16)

def main():
    build_datasets()
    evaluations()

if __name__ == '__main__':
    main()
