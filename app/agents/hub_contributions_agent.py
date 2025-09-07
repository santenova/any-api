from .agent_base import AgentBase
import subprocess
import argparse
import sys
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, NamedTuple
import json
import os
from ..utils.logger import logger

class CommitInfo(NamedTuple):
    """Structure to hold commit information"""
    hash: str
    author: str
    date: str
    datetime: datetime
    lines_added: int
    lines_deleted: int
    files_changed: int
    commit_message: str
    is_merge: bool
    is_likely_import: bool



class HubContributionsAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="HubContributionsAgent", max_retries=max_retries, verbose=verbose)




    def is_likely_import_commit(self, commit_message: str, lines_added: int, lines_deleted: int) -> bool:
      """Detect commits that are likely imports/bootstraps"""
      import_indicators = [
          'import', 'initial', 'bootstrap', 'setup', 'scaffold',
          'copy from', 'migrate from', 'port from', 'add existing',
          'bulk add', 'mass import', 'initial commit'
      ]

      # High line count with low deletions often indicates imports
      if lines_added > 5000 and lines_deleted < lines_added * 0.1:
          return True

      # Check commit message for import indicators
      message_lower = commit_message.lower()
      return any(indicator in message_lower for indicator in import_indicators)


    def run_git_command(self,repo_path: str, command: List[str]) -> str:
      """Run a git command in the specified repository"""
      try:
          result = subprocess.run(
              ['git', '-C', repo_path] + command,
              capture_output=True,
              text=True,
              check=True
          )
          return result.stdout.strip()
      except subprocess.CalledProcessError as e:
          print(f"Error running git command: {e}")
          sys.exit(1)


    def get_commit_data(self,repo_path: str) -> List[CommitInfo]:
      """Extract commit data with better classification"""
      print("🔍 Extracting commit data...")

      # Get comprehensive commit info
      log_format = "--pretty=format:%H|%an|%ad|%s|%P"
      date_format = "--date=format:%Y-%m-%d %H:%M:%S"


      log_output = self.run_git_command(repo_path, [
          'log', log_format, date_format
      ])

      commits = []
      import_commits = 0
      merge_commits = 0
      #print(log_output.strip().split('\n'))

      for line in log_output.strip().split('\n'):

          if not line:
              continue

          parts = line.split('|')
          if len(parts) < 4:
              continue

          commit_hash, author, datetime_str, message = parts[0], parts[1], parts[2], parts[3]
          parents = parts[4] if len(parts) > 4 else ""

          # Parse datetime
          try:
              commit_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
          except ValueError as e:
              logger.error(f"ValueError get_commit_data: {str(e)}")
              continue

          # Check if merge commit
          is_merge = len(parents.split()) > 1
          if is_merge:
              merge_commits += 1

          # Get file changes
          try:
              stat_output = self.run_git_command(repo_path, [
                  'show', '--numstat', '--format=', commit_hash
              ])

              lines_added = 0
              lines_deleted = 0
              files_changed = 0

              for stat_line in stat_output.strip().split('\n'):
                  if not stat_line:
                      continue
                  stat_parts = stat_line.split('\t')
                  if len(stat_parts) >= 2:
                      try:
                          added = int(stat_parts[0]) if stat_parts[0] != '-' else 0
                          deleted = int(stat_parts[1]) if stat_parts[1] != '-' else 0
                          lines_added += added
                          lines_deleted += deleted
                          files_changed += 1
                      except ValueError as e:
                          logger.error(f"ValueError get_commit_data: {str(e)}")
                          continue

              # Detect likely imports
              is_import = self.is_likely_import_commit(message, lines_added, lines_deleted)
              if is_import:
                  import_commits += 1

              commit_info = CommitInfo(
                  hash=commit_hash,
                  author=author,
                  date=commit_datetime.strftime('%Y-%m-%d'),
                  datetime=commit_datetime,
                  lines_added=lines_added,
                  lines_deleted=lines_deleted,
                  files_changed=files_changed,
                  commit_message=message,
                  is_merge=is_merge,
                  is_likely_import=is_import
              )
              commits.append(commit_info)

          except Exception as e:
              logger.error(f"Error get_commit_data: {str(e)}")
              continue

      print(f"📊 Processed {len(commits)} commits")
      print(f"🔀 Merge commits: {merge_commits}")
      print(f"📥 Likely import commits: {import_commits}")
      return commits


    def analyze_contribution_patterns(self,commits: List[CommitInfo]) -> Dict[str, Dict]:
        """Analyze contribution patterns using multiple metrics"""
        print("📊 Analyzing contribution patterns...")

        author_stats = defaultdict(lambda: {
            'total_commits': 0,
            'development_commits': 0,  # Excluding merges and imports
            'lines_added': 0,
            'lines_deleted': 0,
            'files_touched': set(),
            'import_commits': 0,
            'merge_commits': 0,
            'first_commit': None,
            'last_commit': None,
            'active_days': set(),
            'commit_messages': [],
            'monthly_activity': defaultdict(int)
        })

        for commit in commits:
            stats = author_stats[commit.author]
            stats['total_commits'] += 1
            stats['lines_added'] += commit.lines_added
            stats['lines_deleted'] += commit.lines_deleted
            stats['files_touched'].add(commit.hash)  # Proxy for unique files
            stats['active_days'].add(commit.date)
            stats['commit_messages'].append(commit.commit_message)

            # Monthly activity
            month_key = commit.datetime.strftime('%Y-%m')
            stats['monthly_activity'][month_key] += 1

            # Track date range
            if stats['first_commit'] is None or commit.datetime < stats['first_commit']:
                stats['first_commit'] = commit.datetime
            if stats['last_commit'] is None or commit.datetime > stats['last_commit']:
                stats['last_commit'] = commit.datetime

                    # Classify commit types
            if commit.is_merge:
                stats['merge_commits'] += 1
            elif commit.is_likely_import:
                stats['import_commits'] += 1
            else:
                stats['development_commits'] += 1

        # Convert sets to counts and calculate development lines
        for author in author_stats:
            stats = author_stats[author]
            stats['unique_files_estimate'] = len(stats['files_touched'])
            stats['active_days_count'] = len(stats['active_days'])
            stats['files_touched'] = len(stats['files_touched'])  # Convert to count

            # Calculate development lines (excluding imports/merges)
            dev_lines = 0
            for commit in commits:
                if (commit.author == author and
                    not commit.is_merge and
                    not commit.is_likely_import):
                    dev_lines += commit.lines_added + commit.lines_deleted
            stats['development_lines'] = dev_lines

        return dict(author_stats)

    def detect_commit_patterns(self,author_stats: Dict[str, Dict]) -> Dict[str, Dict]:
        """Detect commit patterns to adjust estimation strategies"""

        for author, stats in author_stats.items():
            dev_commits = stats['development_commits']
            dev_lines = stats['development_lines']
            active_days = stats['active_days_count']

            # Calculate pattern indicators
            avg_lines_per_commit = dev_lines / dev_commits if dev_commits > 0 else 0
            commits_per_active_day = dev_commits / active_days if active_days > 0 else 0

            # Detect patterns
            is_large_commit_pattern = avg_lines_per_commit > 200  # Large commits
            is_infrequent_pattern = commits_per_active_day < 1.5  # Less than 1.5 commits per day
            is_squash_heavy = dev_commits < active_days * 0.8  # Fewer commits than 80% of active days

            # Classify developer pattern
            pattern_type = "regular"
            if is_large_commit_pattern and is_infrequent_pattern:
                pattern_type = "large_infrequent"
            elif is_squash_heavy:
                pattern_type = "squash_heavy"
            elif avg_lines_per_commit > 300:
                pattern_type = "bulk_developer"

            stats['pattern_type'] = pattern_type
            stats['avg_lines_per_commit'] = round(avg_lines_per_commit, 1)
            stats['commits_per_active_day'] = round(commits_per_active_day, 1)
            stats['is_large_commit_pattern'] = is_large_commit_pattern
            stats['is_infrequent_pattern'] = is_infrequent_pattern
            stats['is_squash_heavy'] = is_squash_heavy

        return author_stats

    def calculate_alternative_metrics(self,author_stats: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate alternative contribution metrics with pattern-aware adjustments"""
        print("🎯 Calculating alternative metrics...")

        # First detect patterns
        author_stats = self.detect_commit_patterns(author_stats)

        results = {}

        for author, stats in author_stats.items():
            # Calculate time span
            if stats['first_commit'] and stats['last_commit']:
                time_span = stats['last_commit'] - stats['first_commit']
                months_active = time_span.days / 30.44  # Average month length
            else:
                months_active = 0

            # Development intensity (excluding imports/merges)
            dev_commits = stats['development_commits']
            dev_lines = stats['development_lines']
            pattern_type = stats['pattern_type']

            # Estimate based on different methodologies with pattern adjustments
            estimates = {}

            # Method 1: Commits-based with pattern adjustments
            if dev_commits > 0:
                if pattern_type == "large_infrequent":
                    # Large infrequent commits likely represent more work
                    estimates['commits_based_low'] = dev_commits * 4
                    estimates['commits_based_high'] = dev_commits * 12
                elif pattern_type == "squash_heavy":
                    # Squashed commits represent multiple work sessions
                    estimates['commits_based_low'] = dev_commits * 3
                    estimates['commits_based_high'] = dev_commits * 8
                elif pattern_type == "bulk_developer":
                    # Very large commits, likely batch work
                    estimates['commits_based_low'] = dev_commits * 5
                    estimates['commits_based_high'] = dev_commits * 15
                else:
                    # Regular pattern
                    estimates['commits_based_low'] = dev_commits * 1
                    estimates['commits_based_high'] = dev_commits * 3

            # Method 2: Active days with pattern adjustments
            if stats['active_days_count'] > 0:
                if pattern_type in ["large_infrequent", "squash_heavy", "bulk_developer"]:
                    # These patterns suggest more intensive work sessions
                    estimates['days_based_low'] = stats['active_days_count'] * 4
                    estimates['days_based_high'] = stats['active_days_count'] * 10
                else:
                    # Regular pattern
                    estimates['days_based_low'] = stats['active_days_count'] * 2
                    estimates['days_based_high'] = stats['active_days_count'] * 6

            # Method 3: Lines-based with pattern adjustments
            if dev_lines > 0:
                if pattern_type == "large_infrequent":
                    # Large commits may include more thoughtful, complex work
                    estimates['lines_based_low'] = dev_lines / 40  # Slower pace
                    estimates['lines_based_high'] = dev_lines / 15  # But still productive
                elif pattern_type == "squash_heavy":
                    # Squashed commits hide incremental work
                    estimates['lines_based_low'] = dev_lines / 35
                    estimates['lines_based_high'] = dev_lines / 18
                elif pattern_type == "bulk_developer":
                    # May include more copy-paste or generated code
                    estimates['lines_based_low'] = dev_lines / 60
                    estimates['lines_based_high'] = dev_lines / 25
                else:
                    # Regular pattern
                    estimates['lines_based_low'] = dev_lines / 50
                    estimates['lines_based_high'] = dev_lines / 20

            # Method 4: Time-span based (new method for infrequent committers)
            if months_active > 0 and pattern_type in ["large_infrequent", "squash_heavy"]:
                # For infrequent committers, estimate based on sustained activity
                # Assume 10-30 hours per month of active contribution
                estimates['timespan_based_low'] = months_active * 10
                estimates['timespan_based_high'] = months_active * 30

            # Calculate averages with pattern-aware weighting
            low_estimates = [v for k, v in estimates.items() if 'low' in k]
            high_estimates = [v for k, v in estimates.items() if 'high' in k]

            if pattern_type in ["large_infrequent", "squash_heavy", "bulk_developer"]:
                # For these patterns, weight the higher estimates more heavily
                # as they likely underestimate actual work
                avg_low = sum(low_estimates) / len(low_estimates) if low_estimates else 0
                avg_high = sum(high_estimates) / len(high_estimates) if high_estimates else 0
                # Bias toward the higher end for these patterns
                weighted_avg = (avg_low * 0.3 + avg_high * 0.7)
            else:
                # Regular pattern - use simple average
                avg_low = sum(low_estimates) / len(low_estimates) if low_estimates else 0
                avg_high = sum(high_estimates) / len(high_estimates) if high_estimates else 0
                weighted_avg = (avg_low + avg_high) / 2

            results[author] = {
                **stats,
                'months_active': round(months_active, 1),
                'commits_per_month': round(dev_commits / months_active, 1) if months_active > 0 else 0,
                'lines_per_commit': round(dev_lines / dev_commits, 1) if dev_commits > 0 else 0,
                'estimated_hours_low': round(avg_low, 1),
                'estimated_hours_high': round(avg_high, 1),
                'estimated_hours_avg': round((avg_low + avg_high) / 2, 1),
                'estimated_hours_weighted': round(weighted_avg, 1),
                'development_lines': dev_lines,
                'import_percentage': round(stats['import_commits'] / stats['total_commits'] * 100, 1) if stats['total_commits'] > 0 else 0
            }

        return results

    def print_alternative_report(self,results: Dict[str, Dict]):
        """Print a more realistic contribution report"""
        print("\n" + "="*100)
        print("📈 GIT CONTRIBUTION ANALYSIS")
        print("="*100)

        print("\n🎯 Approach:")
        print("  • Uses multiple estimation methods instead of unreliable commit timing")
        print("  • Separates development work from imports/merges")
        print("  • Provides ranges rather than false precision")
        print("  • Considers typical developer productivity rates")

        # Sort by development commits
        sorted_authors = sorted(results.items(),
                              key=lambda x: x[1]['development_commits'],
                              reverse=True)

        print(f"\n{'AUTHOR':<20} {'DEV COMMITS':<11} {'MONTHS':<7} {'PATTERN':<15} {'EST HOURS':<15} {'WEIGHTED':<9}")
        print("-" * 85)

        for author, stats in sorted_authors:
            hours_range = f"{stats['estimated_hours_low']}-{stats['estimated_hours_high']}"
            print(f"{author:<20} {stats['development_commits']:<11} "
                  f"{stats['months_active']:<7} {stats['pattern_type']:<15} "
                  f"{hours_range:<15} {stats['estimated_hours_weighted']:<9}")

        print("\n" + "="*100)
        print("📊 DETAILED ANALYSIS")
        print("="*100)

        #for author, stats in sorted_authors:
        #  print(f"\n👤 {author} {stats}")


    def execute(self, repo_name):
        system_message = "You are an expert software testing and quality assurance agent."
        user_content = f"Analyze test coverage and quality for the repository: {repo_name}\n\n"
        user_content += "Provide insights on:\n"
        user_content += "1. Current test coverage\n"
        user_content += "2. Recommended additional test cases\n"
        user_content += "3. Potential areas of improvement\n"
        user_content += "4. Testing strategy and best practices\n"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        commits = self.get_commit_data(repo_name)

        #print(commits)


        # Analyze patterns
        author_stats = self.analyze_contribution_patterns(commits)
        #print(author_stats)

        # Calculate alternative metrics
        results = self.calculate_alternative_metrics(author_stats)

        # Print report
        self.print_alternative_report(results)
        #print(results)
        return [results]

        #contribution_analysis = self.call_llama(messages, max_tokens=1000)
        #return contribution_analysis

