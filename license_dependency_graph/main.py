import os
import sys
import json
import xml.etree.ElementTree as ET
import tarfile
import zipfile
import email
from collections import defaultdict
from pkg_resources import parse_version

import requests

PACKAGES_DIR = './packages'
if not os.path.isdir(PACKAGES_DIR):
    os.makedirs(PACKAGES_DIR)


CMP_OPERATORS = ['>=', '<=', '==', '<', '>', '!=']


def _split_package_name(package_name):
    """
    example:
        >>> _split_package_name('mindsdb<=1.17.1>1.15.1!=1.15.2')
        >>> ('mindsdb', ('<=', '1.17.1'), ('>', '1.15.1'), ('!=', '1.15.2'))
    """
    s = package_name
    L = []
    for i in reversed(range(len(package_name))):
        for op in CMP_OPERATORS:
            if package_name.startswith(op, i):
                version = s[i + len(op):]
                L.append((op, version))
                s = s[:i]
                break
    return (s, *L)


def _iter_pypi(package_name, cache=dict()):
    if package_name not in cache:
        res = requests.get('https://pypi.org/simple/{}'.format(package_name))
        try:
            res.raise_for_status()
        except Exception:
            cache[package_name] = []
        else:
            tree = ET.fromstring(res.text)
            cache[package_name] = list(tree.findall('.//a'))

    for a in cache[package_name]:
        link = a.get('href')
        link = link.split('#')[0].rstrip('/')
        if link.startswith('/simple'):
            link = 'https://pypi.org' + link
        for ext in ['.tar.gz', '.zip', '.whl']:
            if link.endswith(ext):
                version = link.split('-')[1].rstrip(ext)
                yield (ext, version, link)
                break
        else:
            continue


def _get_license_and_requirements(url, cache=dict()):
    dist_name = url.split('/')[-1]
    dist_path = os.path.join(PACKAGES_DIR, dist_name)

    if not os.path.isfile(dist_path):
        res = requests.get(url)
        res.raise_for_status()
        with open(dist_path, 'wb') as f:
            f.write(res.content)

    if dist_name.endswith('.tar.gz'):
        with tarfile.open(dist_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('PKG-INFO'):
                    parser = email.parser.HeaderParser()
                    pkg_info = tar.extractfile(member).read().decode()
                    license_name = parser.parsestr(pkg_info)['License']
                    break
            else:
                license_name = 'failed to find'
                #raise Exception('failed to find PKG-INFO in {}'.format(dist_name))

            requirements = []
            for member in tar.getmembers():
                if member.name.endswith('requires.txt') or member.name.endswith('requirements.txt'):
                    lines = tar.extractfile(member).read().decode().split('\n')
                    requirements = []
                    for l in lines:
                        # this is temporary
                        if l.startswith('['):
                            break
                        else:
                            requirements.append(l)
                    break
            else:
                pass
                #print('Warning: failed to find requires.txt/requirements.txt in {}'.format(dist_name))

    elif dist_name.endswith('.zip') or dist_name.endswith('.whl'):
        with zipfile.ZipFile(dist_path) as zip_:
            for member in zip_.namelist():
                if member.endswith('metadata.json'):
                    metadata_str = zip_.read(member).decode()
                    metadata = json.loads(metadata_str)
                    if 'license' in metadata:
                        license_name = metadata['license']
                    elif 'License' in metadata:
                        license_name = metadata['License']
                    else:
                        license_name = 'Unknown (failed to find in metadata)'
                    break
            else:
                license_name = 'Unknown (failed to find metadata)'
                #print('Warning: failed to find metadata.json in {}'.format(dist_name))
            
            requirements = []
    else:
        raise Exception('exptected tar/zip/whl')
    
    return (license_name, requirements)


def dep_graph(package_name, D, white_list=None, branch_cache=dict(), visited_chain=list(),):
    print('[{}]'.format(' -> '.join(visited_chain)))
    name, *versions = _split_package_name(package_name)

    try:
        # gropup by version
        releases = defaultdict(list)
        for ext, version, link in _iter_pypi(name):
            releases[version].append((ext, link))

        # sort by extension priority
        ext_priority = ['.tar.gz', '.zip', '.whl']
        for k in releases:
            releases[k] = sorted(
                releases[k],
                key=lambda x: ext_priority.index(x[0])
            )

    except Exception as e:
        print('Warning:', e)
        return

    # filter out versions that we don't need
    to_drop = []
    if len(versions) > 0:
        for release_v in releases:
            for op, v in versions:
                if eval('parse_version("{}") {} parse_version("{}")'.format(release_v, op, v)):
                    break
            else:
                # if none of the conditions was satisfied, drop the relese version
                # NOTE: dict.pop right here will raise RuntimeError
                to_drop.append(release_v)

    for release_v in to_drop:
        releases.pop(release_v, None)

    # extract license and requirements from filtered releases
    for release_v, distributions in releases.items():
        # for now use first distribution
        # later maybe try other in case of failure to get
        # required info from the first distribution
        ext, link = distributions[0]

        key = str((name, release_v))

        if white_list is not None and name in white_list:
            D[key] = 'WhiteListedBranch'
        elif name in visited_chain:
            D[key] = 'CircularDependency'
        else:
            if key in branch_cache:
                D[key] = branch_cache[key]
            else:
                license_name, requirements = _get_license_and_requirements(link)
                D[key] = {'license': license_name, 'requirements': dict()}
                branch_cache[key] = D[key]
                for req in requirements:
                    dep_graph(
                        package_name=req,
                        D=D[key]['requirements'],
                        branch_cache=branch_cache,
                        visited_chain=[*visited_chain, name],
                        white_list=white_list
                    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python ./{} package_name'.format(__file__))

    D = dict()
    dep_graph(sys.argv[1], D, white_list=['nose'])
    with open('out.json', 'w') as f:
        f.write(json.dumps(D))