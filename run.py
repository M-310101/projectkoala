import articlelib, os, time
from pathlib import Path

def test(article_in):
    writeup = Path(__file__).parent / "final results.txt"
    f = open(writeup, 'a')
    tic = time.perf_counter()
    title, text = articlelib.get_article_text(article_in[1])
    f.write(article_in[0])
    f.write('\n')
    if (title == None or text == None):
        f.write("Article does not meet word count minimum.")
        f.write('\n')
        toc = time.perf_counter()
        elapsed = toc - tic
        str_out = 'Time taken: ' + str(elapsed)
        f.write(str_out)
        f.write('\n\n')
        f.close()
        return

    # process text and get stemmed version
    model_text = articlelib.text_preprocess_stem(text)

    # use our fake news model to predict whether the article is fake news or not
    # 1 for fake news, 0 for real news
    reliability_model_score = articlelib.fn_model.predict([model_text])[0]
    if (reliability_model_score == 0):
        f.write('Article passes fake news model')
        f.write('\n')
    else:
        f.write('Article did not pass fake news model')
        f.write('\n')
    #     f.write("Based on a dataset of over 65,000 articles, this article is predicted to be reliable.")
    #     f.write('\n\n')
    #     f.close()
    #     return
    # else:
    #     print("Article did not pass the article reliability model, comparing against online resources\n")

    comp_text = articlelib.nlp(articlelib.text_preprocess_lem(text))
    text_ents = [x.text for x in comp_text.ents]
    set_len = len(text_ents)

    # get sentiment for inputted article
    sentiment = articlelib.text_sentiment(text)

    found_articles = {
        'lem_text':[],
        'sent_neg':[],
        'sent_neu':[],
        'sent_pos':[],
        'sent_comp':[],
        'text_ents':[],
        }



    '''Get sites from single article checker'''

    # get a list of url's based off a query search on google
    gathered_sites = articlelib.gather_sites(title)
    num_of_compares = sum_sim = 0
    str_out = 'Retreived ' + str(len(gathered_sites)) + ' URLs'
    f.write(str_out)
    f.write('\n')

    # get the text from each site and compare similarity score
    for site in gathered_sites:
        found_text = articlelib.gather_content(site)

        # if the response from getting content from url is not text, skip
        if (type(found_text) != str):
            continue
        found_comp_lem = articlelib.text_preprocess_lem(found_text)
        if (found_comp_lem == None):
            continue
        found_comp_text = articlelib.nlp(articlelib.text_preprocess_lem(found_text))
        sim_score = comp_text.similarity(found_comp_text)
        # if (sim_score >= 0.97):

        # check similarity score, skip article if below 0.25
        if (sim_score < 0.25):
            continue

        num_of_compares += 1
        sum_sim += sim_score

        # fill dictionary with entry information
        found_articles['lem_text'].append(found_comp_text)
        found_sentiment = articlelib.text_sentiment(found_text)
        found_articles['sent_pos'].append(found_sentiment['pos'])
        found_articles['sent_neu'].append(found_sentiment['neu'])
        found_articles['sent_neg'].append(found_sentiment['neg'])
        found_articles['sent_comp'].append(found_sentiment['compound'])
        found_ents = [x.text for x in found_comp_text.ents]
        found_articles['text_ents'].append(found_ents)

    # if there are no valid articles found to compare with, exit
    if (num_of_compares == 0):
        f.write("Failed to find any similar articles, cannot come to any conculsions.")
        f.write('\n')
        toc = time.perf_counter()
        elapsed = toc - tic
        str_out = 'Time taken: ' + str(elapsed)
        f.write(str_out)
        f.write('\n\n')
        f.close()
        return

    str_out = 'Found ' + str(num_of_compares) + ' comparable articles'
    f.write(str_out)
    f.write('\n')

    '''Get similarity score based on found articles'''

    # get average similarity score
    avr_sim = sum_sim / num_of_compares
    if (avr_sim >= 0.972):
        f.write('Found articles on same topic shared a high similarity')
        f.write('\n')
    #     f.write("Found articles on same topic shared a high similarity to this article, can be considered reliable.")
    #     f.write('\n\n')
    #     f.close()
    #     return
    else:
        f.write('Found articles on same topic did not share a high similarity')
        f.write('\n')


    '''If there is still uncertainty on article reliability, calculate independent reliability scores'''

    # article neutrality threshold
    neutraility = True
    # threshold for neutrality score
    if (sentiment['neu'] < 0.82):
        neutraility = False


    # get bucket value
    bucket_val, sent_str, degree_str = articlelib.get_bucket((sentiment['pos'] - sentiment['neg']))
    
    # calculate average sentiment difference score between other articles
    total_diff_score = 0
    for i in range(num_of_compares):
        comp_bucket_val, *_ = articlelib.get_bucket((found_articles['sent_pos'][i] - found_articles['sent_neg'][i]))

        # get the score as a value between 0 - 1
        total_diff_score += (abs(bucket_val - comp_bucket_val) / 8)

    avr_diff_score = total_diff_score/num_of_compares
    str_out = 'Average sentiment difference score: ' + str(avr_diff_score)
    f.write(str_out)
    f.write('\n')
    # calculate as a score between 0 to 100
    mapped_sentiment_score = articlelib.map_sent_score(avr_diff_score, 100)


    # compare fact set from each
    total_set_score = 0
    for i in range(num_of_compares):
        comp_ents = found_articles['text_ents'][i]
        low_len = min(set_len, len(comp_ents))
        if (low_len == 0):
            continue
        num_shared_set = len(set(text_ents).intersection(comp_ents))
        shared_ratio = num_shared_set / low_len
        total_set_score += shared_ratio

    set_score = total_set_score/num_of_compares

    str_out = 'Average fact match score: ' + str(set_score)

    # transpose score to number between 0 and 100
    mapped_set_score = articlelib.map_set_score(set_score, 100)


    str_out = 'Article is ' + degree_str + ' ' + sent_str
    f.write(str_out)

    f.write('\n')
    str_out = 'Neutraility: ' + str(neutraility)
    f.write(str_out)

    f.write('\n')
    str_out = 'Mapped sentiment matching score: ' + str(mapped_sentiment_score)
    f.write(str_out)

    f.write('\n')
    str_out = 'Mapped fact set matching score: ' + str(mapped_set_score)
    f.write(str_out)

    f.write('\n')
    toc = time.perf_counter()
    elapsed = toc - tic
    str_out = 'Time taken: ' + str(elapsed)
    f.write(str_out)

    f.write('\n\n')
    f.close()
    return













def main():

    # url = 'https://www.irishtimes.com/business/economy/world/jp-morgan-in-record-13-billion-settlement-with-us-authorities-1.1601097'
    # print(url)
    # james = articlelib.gather_content(url)
    # if (james != None):
    #     print(james)

    # # sentiment = articlelib.text_sentiment(james)
    # # print(sentiment)
    # # print(type(sentiment))
    # old = articlelib.text_preprocess_stem(article)
    # print(old)
    # print('\n\n\n')

    # new = articlelib.text_preprocess_lem(article)
    # print(new)









    '''Populate variables with article data'''
    # # load in article
    # loc = Path(__file__).parent / "article.txt"
    # title, text = articlelib.get_article_text(loc)
    # if (title == None or text == None):
    #     print("Article does not meet word count minimum.")
    #     return

    # # process text and get stemmed version
    # model_text = articlelib.text_preprocess_stem(text)

    # # use our fake news model to predict whether the article is fake news or not
    # # 1 for fake news, 0 for real news
    # reliability_model_score = articlelib.fn_model.predict([model_text])[0]
    # if (reliability_model_score == 0):
    #     print("Based on a dataset of over 65,000 articles, this article is predicted to be reliable.")
    #     # return
    # else:
    #     print("Article did not pass the article reliability model, comparing against online resources\n")

    # comp_text = articlelib.nlp(articlelib.text_preprocess_lem(text))
    # text_ents = [x.text for x in comp_text.ents]
    # set_len = len(text_ents)

    # # get sentiment for inputted article
    # sentiment = articlelib.text_sentiment(text)

    # found_articles = {
    #      'lem_text':[],
    #      'sent_neg':[],
    #      'sent_neu':[],
    #      'sent_pos':[],
    #      'sent_comp':[],
    #      'text_ents':[],
    #      }



    # '''Get sites from single article checker'''

    # # get a list of url's based off a query search on google
    # gathered_sites = articlelib.gather_sites(title)
    # num_of_compares = sum_sim = 0

    # # get the text from each site and compare similarity score
    # for site in gathered_sites:
    #     found_text = articlelib.gather_content(site)

    #     # if the response from getting content from url is not text, skip
    #     if (type(found_text) != str):
    #         continue
    #     found_comp_text = articlelib.nlp(articlelib.text_preprocess_lem(found_text))
    #     sim_score = comp_text.similarity(found_comp_text)

    #     # check similarity score, skip article if below 0.5
    #     if (sim_score < 0.5):
    #         continue

    #     num_of_compares += 1
    #     sum_sim += sim_score

    #     # fill dictionary with entry information
    #     found_articles['lem_text'].append(found_comp_text)
    #     found_sentiment = articlelib.text_sentiment(found_text)
    #     found_articles['sent_pos'].append(found_sentiment['pos'])
    #     found_articles['sent_neu'].append(found_sentiment['neu'])
    #     found_articles['sent_neg'].append(found_sentiment['neg'])
    #     found_articles['sent_comp'].append(found_sentiment['compound'])
    #     found_ents = [x.text for x in found_comp_text.ents]
    #     found_articles['text_ents'].append(found_ents)

    # # if there are no valid articles found to compare with, exit
    # if (num_of_compares == 0):
    #     print("Failed to find any similar articles, cannot come to any conculsions.")
    #     return


    # '''Get similarity score based on found articles'''

    # # get average similarity score
    # avr_sim = sum_sim / num_of_compares
    # if (avr_sim >= 0.972):
    #     print("Found articles on same topic shared a high similarity to this article, can be considered reliable.")
    #     return


    # '''If there is still uncertainty on article reliability, calculate independent reliability scores'''

    # # article neutrality threshold
    # neutraility = True
    # # threshold for neutrality score
    # if (sentiment['neu'] < 0.82):
    #     neutraility = False


    # # get bucket value
    # bucket_val, sent_str, degree_str = articlelib.get_bucket((sentiment['pos'] - sentiment['neg']))
    
    # # calculate average sentiment difference score between other articles
    # total_diff_score = 0
    # for i in range(num_of_compares):
    #     comp_bucket_val, *_ = articlelib.get_bucket((found_articles['sent_pos'][i] - found_articles['sent_neg'][i]))

    #     # get the score as a value between 0 - 1
    #     total_diff_score += (abs(bucket_val - comp_bucket_val) / 8)

    # avr_diff_score = total_diff_score/num_of_compares
    # # calculate as a score between 0 to 100
    # mapped_sent_score = articlelib.map_sent_score(avr_diff_score, 100)


    # # compare fact set from each
    # total_set_score = 0
    # for i in range(num_of_compares):
    #     comp_ents = found_articles['text_ents'][i]
    #     low_len = min(set_len, len(comp_ents))
    #     num_shared_set = len(set(text_ents).intersection(comp_ents))
    #     shared_ratio = num_shared_set / low_len
    #     total_set_score += shared_ratio

    # set_score = total_set_score/num_of_compares

    # # transpose score to number between 0 and 100
    # mapped_set_score = articlelib.map_set_score(set_score, 100)




    # print('Article is', degree_str, sent_str)
    # print('Neutraility:', neutraility)
    # print('Sentiment matching score:', mapped_sent_score)
    # print('Fact set matching score:', mapped_set_score)















    '''run for all test articles'''
    # article_list = articlelib.get_article_list()
    # for i in range(0, len(article_list)):
    #     test(article_list[i])
    article_list = articlelib.get_fake_article_list()
    for i in range(0, len(article_list)):
        test(article_list[i])

    # for i in range(0, 60):
    #     # reset similarity scores
    #     num_of_found = sum_score = 0
    #     # get article n
    #     title, text = articlelib.get_article_text(article_list[i][1])
    #     if (title == None or text == None):
    #         continue

    #     comp_text = articlelib.nlp(articlelib.text_preprocess_lem(text))

    #     # get list of related sites
    #     sites = articlelib.gather_sites(title)
    #     for site in sites:

    #         found_text = articlelib.gather_content(site)

    #         # if the response from getting content from url is not text, skip
    #         if (type(found_text) != str):
    #             continue
    #         found_comp_text = articlelib.nlp(articlelib.text_preprocess_lem(found_text))
    #         sim_score = comp_text.similarity(found_comp_text)

    #         # if the similarity score is below this threshold, very unlikely that the articles are related
    #         # or even useful for comparison
    #         if (sim_score < 0.5):
    #             continue
            
    #         num_of_found += 1
    #         sum_score += sim_score

    #     # get average similarity score        
    #     avr_sim = sum_score/num_of_found

    #     line = 'Average similarity score: ' + str(avr_sim)
    #     print(title)
    #     print('Found', num_of_found, 'usable sites.')
    #     print(line)
    #     print('\n\n')

    return











if __name__ == '__main__':
    main()

