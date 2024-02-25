import requests
import time
import matplotlib.pyplot as plt
import pandas as pd

owner = 'karenaliang'
repo = 'personalwebsite'
token = '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'  


print("=============")



##### ALL REPOSITORIES #####

## Total Commits in Github ##

# prompt user to enter info
# owner = input("Enter your GitHub username: ")
# token = input("Enter access token: ")

# get events from GitHub API
response = requests.get(f"https://api.github.com/users/{owner}/events")

# check if request was successful 
if response.status_code == 200:
    # count the occurrences of "PushEvent" in the response text
    commit_count = response.text.count("PushEvent")
    # print the total commit count
    print(f"Total Commit Count: {commit_count}")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")


print("=============")


## Language Breakdown in Github ##
print("Language Breakdown in Github")

repos_url = f"https://api.github.com/users/{owner}/repos"

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github.v3+json"
}

# fetching all user repos
response = requests.get(repos_url, headers=headers)
repositories = response.json()
languages_df = pd.DataFrame()     # dataframe with all languages and their bytes b4 aggregation

if response.status_code == 200:
    for repo in repositories:
        ## per every repo ##
        repo_name = repo["name"]
        languages_url = f"https://api.github.com/repos/{owner}/{repo_name}/languages"
        response = requests.get(languages_url, headers=headers)

        
        # checking again for language request #
        if response.status_code == 200:
            languages_data = response.json()
            total_bytes = sum(languages_data.values())

            # print(f"Languages used in repository '{repo_name}': {languages_data}")
            percents = languages_data.copy()          
            percent_keys = percents.keys()

            for key in percent_keys:
                percents[key] = percents.get(key)/total_bytes

            repo_df = pd.DataFrame(list(languages_data.items()), columns=['Language', 'Bytes'])
            # print(repo_df)
            # print(languages_df)
            # print(" << after >> ")
            languages_df = pd.concat([languages_df, repo_df], ignore_index=True)
            # print(languages_df)
            # print("==============================")

            # print(f"Percent Usage Per Language: '{repo_name}': {percents}")
        else:
            print(f"Failed to fetch languages for repository '{repo_name}'")


    ## Language Percentages in Total (All Repos Aggregated) ##
    agg_bytes = languages_df.groupby("Language").agg("sum")
    total_bytes = agg_bytes['Bytes'].sum()
    percent_bytes = agg_bytes.copy()
    percent_bytes['Bytes'] = percent_bytes['Bytes'].astype(float)

    for i in agg_bytes.index:
        percent_bytes.at[i, 'Bytes'] = agg_bytes.at[i, 'Bytes'] / total_bytes


    ## Language Percentages Based on Count ##
    lang_ct = languages_df.groupby("Language").agg("count")
    total_bytes = lang_ct['Bytes'].sum()

    percent_lang_ct = lang_ct.copy()
    percent_lang_ct['Bytes'] = percent_lang_ct['Bytes'].astype(float)


    for i in lang_ct.index:
        percent_lang_ct.at[i, 'Bytes'] = lang_ct.at[i, 'Bytes'] / total_bytes


    # percent_lang_ct.plot.pie(y='Bytes', figsize=(5, 5))
    # plt.show()

    # ## Plotting Percentages ##
    # p_df = pd.DataFrame.from_dict(percents, orient='index', columns=['Percentage'], dtype=None)
    # p_df.plot.pie(y='Percentage', figsize=(5, 5))
    # # plt.show()

else:
    print(f"Failed to fetch repositories. Status code: {response.status_code}")



print("=============")


##### BY INDIVIDUAL REPOSITORY #####

# ### BASIC REPO STATS ###

# def get_github_repo_stats(owner, repo, token=None):
#     base_url = "https://api.github.com/repos"
#     url = f"{base_url}/{owner}/{repo}"

#     headers = {}
#     if token:
#         headers["Authorization"] = f"Bearer {token}"

#     response = requests.get(url, headers=headers)

#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"Error: {response.status_code}")
#         return None

# repo_stats = get_github_repo_stats(owner, repo, token)

# if repo_stats:
#     print(f"Repository Name: {repo_stats['name']}")
#     print(f"Stars: {repo_stats['stargazers_count']}")
#     print(f"Forks: {repo_stats['forks_count']}")
#     print(f"Watchers: {repo_stats['subscribers_count']}")
#     print(f"Issues: {repo_stats['open_issues_count']}")
#     print(f"Description: {repo_stats['description']}")
# else:
#     print("Failed to retrieve repository stats.")



### Weekly Commit Activity ###

# def get_code_frequency_stats(owner, repo, token=None):
#     base_url = "https://api.github.com/repos"
#     url = f"{base_url}/{owner}/{repo}/stats/code_frequency"

#     headers = {}
#     if token:
#         headers["Authorization"] = f"Bearer {token}"
#         headers["X-GitHub-Api-Version"] = "2022-11-28"

#     response = requests.get(url, headers=headers)

#     if response.status_code == 200:
#         return response.json()
#     elif response.status_code == 202:
#         print("Request accepted. Check again later.")
#         return None
#     elif response.status_code == 204:
#         print("No content.")
#         return None
#     elif response.status_code == 422:
#         print("Repository contains more than 10,000 commits.")
#         return None
#     else:
#         print(f"Error: {response.status_code}")
#         return None

# code_freq = None

# while code_freq is None:
#     code_freq = get_code_frequency_stats(owner, repo)
#     time.sleep(5)  # Check again after 5 seconds

# if code_freq:
#     print("Code Frequency Stats:")
#     print(f"Timestamp: {code_freq[0][0]}")
#     print(f"Additions: {code_freq[0][1]}")
#     print(f"Deletions: {code_freq[0][2]}")
# else:
#     print("Failed to retrieve code frequency stats.")




# def get_weekly_commit_count(owner, repo, token=None):
#     base_url = "https://api.github.com/repos"
#     url = f"{base_url}/{owner}/{repo}/stats/participation"

#     headers = {
#         "Accept": "application/vnd.github.v3+json",
#     }

#     if token:
#         headers["Authorization"] = f"Bearer {token}"
#         headers["X-GitHub-Api-Version"] = "2022-11-28"

#     response = requests.get(url, headers=headers)

#     if response.status_code == 200:
#         return response.json()
#     elif response.status_code == 202:
#         print("Request accepted. Checking again in a moment.")
#         return None
#     elif response.status_code == 204:
#         print("No content.")
#         return None
#     elif response.status_code == 403:
#         print("Error: 403 Forbidden - Authentication or permission issue.")
#         return None
#     else:
#         print(f"Error: {response.status_code}")
#         return None

# commit_ct = None

# while commit_ct is None:
#     commit_ct = get_weekly_commit_count(owner, repo)
#     time.sleep(5)  # Check again after 5 seconds

# if commit_ct:
#     print("Weekly Commit Count (One Year):")
#     print(f"All: {commit_ct['all']}")
#     print(f"Owner: {commit_ct['owner']}")

# else:
#     print("Failed to retrieve code frequency stats.")
