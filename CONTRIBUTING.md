#  Community guideline

* This is intended to be a living document which will evolve over time -
  suggestions are welcome.
* Collaboration on this repository is loosely based on the feature branch
  workflow model - more information can be found on this [here](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow).

## Repository structure

```
dust
|-- Projects
|   |-- ABM_DA
|   |   |-- bussim
|   |   |   |-- ...
|   |   |
|   |   |-- experiments
|   |   |   |-- pf_experiments
|   |   |   |   |-- ...
|   |   |   |
|   |   |   |-- ukf_experiments
|   |   |   |   |-- ...
|   |   |
|   |   |-- stationsim
|   |
|   |-- BusSim
|   |
|   |-- Data_Assimilation_Notebooks
|   |
|   |-- Improbable
|   |
|   |-- Probabilistic_Experiments
|
|-- Writing
```

* The dust repository consists of two main directories: `Projects` and
  `Writing`. The former is where the majority of work resides - specifically
  code that contributes to different investigations; the latter contains writing
  that has been undertaken towards the dissemination of work, including papers
  and blog posts.
* Information on the other subdirectories...

## General workflow

For the majority of your work, you will likely be doing the following:

1. Make local changes to dust on your branch.
2. Commit your changes to your branch.
3. Push changes on your branch.

When you have completed a piece of work on your branch, you will likely want to
make it available to everyone else so that they can start using it and
interacting with it.
We do this by issuing a pull request to merge our branch into the master branch.
When doing this, we specify our branch as the source and master as the
destination.
In order to achieve this, we do the following:

1. Ensure that you have completed the above steps - commit the changes on your
   branch and push them.
2. Open the repository on the Github website.
3. Change to the branch on which you have been working, and click the "New pull
   request" button.
4. You can now compare two branches - the first should be the branch into which
   you would like to merge (i.e. the `master` branch), whilst the other should
   be the branch containing your work which you would like to merge in.
5. Provide a title and some comments for your pull request - these should be
   somewhat descriptive so that people can get a general overview of what the
   pull request encapsulates.
6. On the right bar, add a review (typically Nick), and an assignee (this is
   typically yourself, but you may additionally wish to include someone else who
   has also helped with the code, or who may be working on it as follow-up).
7. Once you're happy with everything, hit the "Create pull request" button!

## General git etitquette

* When creating a new directory or subdirectory, always include a README.md file
  in the directory to let people know what the directory is for.
* When working on your branch, it is a good idea to merge changes from the
  master branch into your branch - this ensures that your branch stays
  up-to-date with everyone else's changes and work when it becomes available.

## Commit etiquette

When getting started with git, it can be easy to forget to stage changes and
commit on a regular basis.
Putting in a little bit of effort, however, can go a long way to ensuring that the
repository is well maintained for both yourself and others.
Below are a few guidelines to help:

* Please try to ensure that your code works before making a commit; remember that each
  commit acts as a checkpoint to which you (and your collaborators) can rewind -
  if someone rewinds to your commit then they will probably expect to find the
  code in a working state.
* Try to avoid too many changes piling up before you make a commit. It is better
  to frequently commit small discrete changes that achieve a clearly defined
  task (e.g. writing a function or a class). This allows us to clearly identify
  what each commit is trying to achieve.
* Please provide short but informative commit messages. Some changes that you
  make may be incredibly minor, such as fixing some typos, but a well written
  commit message will enable anyone reading it to understand what you have
  achieved without having to dig down into the code.

## Branching etiquette

* Please avoid working directly on the `master` branch - this is the branch that
  is on display for viewing by external parties. It is better, instead, to work
  on your own branch which can then be merged into `master` when necessary.
* A suggested naming convention for branches is `dev_{initials}` where
  `{initials}` would be the initials of the person who is working on the branch
  and `feature_{feature}` where `{feature}` would be a specific contained job
  that the branch is trying to complete. The former should be used for general
  work that you are undertaking, and the latter should be used for fixing
  specific issues.
