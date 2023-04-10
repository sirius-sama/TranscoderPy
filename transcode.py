#!/usr/bin/env python3
import os
import re
import sys
import pipes
import errno
import shlex
import cli_ui
import shutil
import signal
import argparse
import subprocess
import mutagen.flac
import multiprocessing
from rich.console import Console

import tagging

console = Console()

encoders = {
    '320':  {'enc': 'lame', 'ext': '.mp3',  'opts': '-h -b 320 --ignore-tag-errors'},
    'V0':   {'enc': 'lame', 'ext': '.mp3',  'opts': '-V 0 --vbr-new --ignore-tag-errors'},
    'V2':   {'enc': 'lame', 'ext': '.mp3',  'opts': '-V 2 --vbr-new --ignore-tag-errors'},
    'FLAC': {'enc': 'flac', 'ext': '.flac', 'opts': '--best'}
}

class TranscodeException(Exception):
    pass

class TranscodeDownmixException(TranscodeException):
    pass

class UnknownSampleRateException(TranscodeException):
    pass
    
# In most Unix shells, pipelines only report the return code of the
# last process. We need to know if any process in the transcode
# pipeline fails, not just the last one.
#
# This function constructs a pipeline of processes from a chain of
# commands just like a shell does, but it returns the status code (and
# stderr) of every process in the pipeline, not just the last one. The
# results are returned as a list of (code, stderr) pairs, one pair per
# process.
def run_pipeline(cmds):
    # The Python executable (and its children) ignore SIGPIPE. (See
    # http://bugs.python.org/issue1652) Our subprocesses need to see
    # it.
    sigpipe_handler = signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    stdin = None
    last_proc = None
    procs = []
    try:
        for cmd in cmds:
            proc = subprocess.Popen(shlex.split(cmd), stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if last_proc:
                # Ensure last_proc receives SIGPIPE if proc exits first
                last_proc.stdout.close()
            procs.append(proc)
            stdin = proc.stdout
            last_proc = proc
    finally:
        signal.signal(signal.SIGPIPE, sigpipe_handler)

    last_stderr = last_proc.communicate()[1]

    results = []
    for (cmd, proc) in zip(cmds[:-1], procs[:-1]):
        # wait() is OK here, despite use of PIPE above; these procs
        # are finished.
        proc.wait()
        results.append((proc.returncode, proc.stderr.read()))
    results.append((last_proc.returncode, last_stderr))
    return results

def locate(root, match_function, ignore_dotfiles=True):
    '''
    Yields all filenames within the root directory for which match_function returns True.
    '''
    for path, dirs, files in os.walk(root):
        for filename in (os.path.abspath(os.path.join(path, filename)) for filename in files if match_function(filename)):
            if ignore_dotfiles and os.path.basename(filename).startswith('.'):
                pass
            else:
                yield filename

def ext_matcher(*extensions):
    '''
    Returns a function which checks if a filename has one of the specified extensions.
    '''
    return lambda f: os.path.splitext(f)[-1].lower() in extensions

def is_24bit(flac_dir):
    '''
    Returns True if any FLAC within flac_dir is 24 bit.
    '''
    flacs = (mutagen.flac.FLAC(flac_file) for flac_file in locate(flac_dir, ext_matcher('.flac')))
    return any(flac.info.bits_per_sample > 16 for flac in flacs)

def is_multichannel(flac_dir):
    '''
    Returns True if any FLAC within flac_dir is multichannel.
    '''
    flacs = (mutagen.flac.FLAC(flac_file) for flac_file in locate(flac_dir, ext_matcher('.flac')))
    return any(flac.info.channels > 2 for flac in flacs)

def needs_resampling(flac_dir):
    '''
    Returns True if any FLAC within flac_dir needs resampling when
    transcoded.
    '''
    return is_24bit(flac_dir)

def resample_rate(flac_dir):
    '''
    Returns the rate to which the release should be resampled.
    '''
    flacs = (mutagen.flac.FLAC(flac_file) for flac_file in locate(flac_dir, ext_matcher('.flac')))
    original_rate = max(flac.info.sample_rate for flac in flacs)
    if original_rate % 44100 == 0:
        return 44100
    elif original_rate % 48000 == 0:
        return 48000
    else:
        return None

def transcode_commands(output_format, resample, needed_sample_rate, flac_file, transcode_file):
    '''
    Return a list of transcode steps (one command per list element),
    which can be used to create a transcode pipeline for flac_file ->
    transcode_file using the specified output_format, plus any
    resampling, if needed.
    '''
    if resample:
        flac_decoder = 'sox {FLAC} -G -b 16 -t wav - rate -v -L {SAMPLERATE} dither'
    else:
        flac_decoder = 'flac -dcs -- {FLAC}'

    lame_encoder = 'lame -S {OPTS} - {FILE}'
    flac_encoder = 'flac {OPTS} -o {FILE} -'

    transcoding_steps = [flac_decoder]

    if encoders[output_format]['enc'] == 'lame':
        transcoding_steps.append(lame_encoder)
    elif encoders[output_format]['enc'] == 'flac':
        transcoding_steps.append(flac_encoder)

    transcode_args = {
        'FLAC' : pipes.quote(flac_file),
        'FILE' : pipes.quote(transcode_file),
        'OPTS' : encoders[output_format]['opts'],
        'SAMPLERATE' : needed_sample_rate,
    }

    if output_format == 'FLAC' and resample:
        commands = ['sox {FLAC} -G -b 16 {FILE} rate -v -L {SAMPLERATE} dither'.format(**transcode_args)]
    else:
        commands = map(lambda cmd: cmd.format(**transcode_args), transcoding_steps)
    return commands

# Pool.map() can't pickle lambdas, so we need a helper function.
def pool_transcode(args):
    return transcode(*args)

def transcode(flac_file, output_dir, output_format):
    '''
    Transcodes a FLAC file into another format.
    '''
    # gather metadata from the flac file
    flac_info = mutagen.flac.FLAC(flac_file)
    sample_rate = flac_info.info.sample_rate
    bits_per_sample = flac_info.info.bits_per_sample
    resample = sample_rate > 48000 or bits_per_sample > 16

    # if resampling isn't needed then needed_sample_rate will not be used.
    needed_sample_rate = None

    if resample:
        if sample_rate % 44100 == 0:
            needed_sample_rate = '44100'
        elif sample_rate % 48000 == 0:
            needed_sample_rate = '48000'
        else:
            raise UnknownSampleRateException('FLAC file "{0}" has a sample rate {1}, which is not 88.2, 176.4, 96, or 192kHz but needs resampling, this is unsupported'.format(flac_file, sample_rate))

    if flac_info.info.channels > 2:
        raise TranscodeDownmixException('FLAC file "{0}" has more than 2 channels, unsupported'.format(flac_file))

    # determine the new filename
    transcode_basename = os.path.splitext(os.path.basename(flac_file))[0]
    transcode_basename = re.sub(r'[\?<>\\*\|"]', '_', transcode_basename)
    transcode_file = os.path.join(output_dir, transcode_basename)
    transcode_file += encoders[output_format]['ext']

    if not os.path.exists(os.path.dirname(transcode_file)):
        try:
            os.makedirs(os.path.dirname(transcode_file))
        except OSError as e:
            if e.errno == errno.EEXIST:
                # Harmless race condition -- another transcode process
                # beat us here.
                pass
            else:
                raise e

    commands = list(transcode_commands(output_format, resample, needed_sample_rate, flac_file, transcode_file))
    results = run_pipeline(commands)

    # Check for problems. Because it's a pipeline, the earliest one is
    # usually the source. The exception is -SIGPIPE, which is caused
    # by "backpressure" due to a later command failing: ignore those
    # unless no other problem is found.
    last_sigpipe = None
    for (cmd, (code, stderr)) in zip(commands, results):
        if code:
            if code == -signal.SIGPIPE:
                last_sigpipe = (cmd, (code, stderr))
            else:
                raise TranscodeException('Transcode of file "{0}" failed: {1}'.format(flac_file, stderr))
    if last_sigpipe:
        # XXX: this should probably never happen....
        raise TranscodeException('Transcode of file "{0}" failed: SIGPIPE'.format(flac_file))

    tagging.copy_tags(flac_file, transcode_file)
    (ok, msg) = tagging.check_tags(transcode_file)
    if not ok:
        raise TranscodeException('Tag check failed on transcoded file: {0}'.format(msg))

    return transcode_file

def path_length_exceeds_limit(flac_dir, basename):
    path_length = 0;
    flac_files = locate(flac_dir, ext_matcher('.flac'))

    source_directory_name = flac_dir[flac_dir.rfind('/') + 1:-1]

    for root, dirs, files in os.walk(flac_dir):
        for name in files:
            if len(basename + root[root.rfind(source_directory_name) + len(source_directory_name) + 1:-1] + "/" + name) > 180:
                    return True
            else:
                pathcheck = len(basename + root[root.rfind(source_directory_name) + len(source_directory_name) + 1:-1] + "/" + name)

    return False

def get_transcode_dir(flac_dir, output_dir, basename, output_format, resample):
    if output_format == "FLAC":
        basename += " [FLAC - Lossless]"
    elif output_format == "V0":
        basename += " [MP3 - V0]"
    elif output_format == "320":
        basename += " [MP3 - 320]"
    else:
        basename += " [MP3 - V2]"
    
    while path_length_exceeds_limit(flac_dir, basename):
        basename = input("The file paths in this torrent exceed the 180 character limit. \n\
        The current directory name is: " + basename + " \n\
        Please enter a shorter directory name: ")

    basename = basename.replace('\\', '').replace('/', '').replace(':', '').replace('*', '').replace('?', '').replace('"', '').replace('<', '').replace('>', '').replace('|', '')
    return os.path.join(output_dir, basename)

def transcode_release(flac_dir, output_dir, basename, output_format, max_threads=None):
    '''
    Transcode a FLAC release into another format.
    '''
    flac_dir = os.path.abspath(flac_dir)
    output_dir = os.path.abspath(output_dir)
    flac_files = locate(flac_dir, ext_matcher('.flac'))

    # check if we need to resample
    resample = needs_resampling(flac_dir)

    # check if we need to encode
    if output_format == 'FLAC' and not resample:
        # XXX: if output_dir is not the same as flac_dir, this may not
        # do what the user expects.
        if output_dir != os.path.dirname(flac_dir):
            print("Warning: no encode necessary, so files won't be placed in", output_dir)
        return flac_dir

    # make a new directory for the transcoded files
    #
    # NB: The cleanup code that follows this block assumes that
    # transcode_dir is a new directory created exclusively for this
    # transcode. Do not change this assumption without considering the
    # consequences!
    console.print(f"[green]Adding {output_format} to...")
    transcode_dir = get_transcode_dir(flac_dir, output_dir, basename, output_format, resample)
    print(transcode_dir)
    if not os.path.exists(transcode_dir):
        os.makedirs(transcode_dir)
    else:
        return transcode_dir
        #raise TranscodeException('transcode output directory "%s" already exists' % transcode_dir)

    # To ensure that a terminated pool subprocess terminates its
    # children, we make each pool subprocess a process group leader,
    # and handle SIGTERM by killing the process group. This will
    # ensure there are no lingering processes when a transcode fails
    # or is interrupted.
    def pool_initializer():
        os.setsid()
        def sigterm_handler(signum, frame):
            # We're about to SIGTERM the group, including us; ignore
            # it so we can finish this handler.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            pgid = os.getpgid(0)
            os.killpg(pgid, signal.SIGTERM)
            sys.exit(-signal.SIGTERM)
        signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        # create transcoding threads
        #
        # Use Pool.map() rather than Pool.apply_async() as it will raise
        # exceptions synchronously. (Don't want to waste any more time
        # when a transcode breaks.)
        #
        # XXX: actually, use Pool.map_async() and then get() the result
        # with a large timeout, as a workaround for a KeyboardInterrupt in
        # Pool.join(). c.f.,
        # http://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool?rq=1
        pool = multiprocessing.Pool(max_threads, initializer=pool_initializer)
        try:
            result = pool.map_async(pool_transcode, [(filename, os.path.dirname(filename).replace(flac_dir, transcode_dir), output_format) for filename in flac_files])
            result.get(60 * 60 * 12)
            pool.close()
        except:
            pool.terminate()
            raise
        finally:
            pool.join()

        # copy other files
        allowed_extensions = ['.cue', '.gif', '.jpeg', '.jpg', '.log', '.md5', '.nfo', '.pdf', '.png', '.sfv', '.txt']
        allowed_files = locate(flac_dir, ext_matcher(*allowed_extensions))
        for filename in allowed_files:
            new_dir = os.path.dirname(filename).replace(flac_dir, transcode_dir)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            shutil.copy(filename, new_dir)

        return transcode_dir

    except:
        # Cleanup.
        #
        # ASSERT: transcode_dir was created by this function and does
        # not contain anything other than the transcoded files!
        shutil.rmtree(transcode_dir)
        raise


def main():
    """
    Setup argparser
    """
    parser = argparse.ArgumentParser(description="hello")
    parser.add_argument('path', help="Path to the directory")
    parser.add_argument('-t', '--threads', nargs='*', required=False, default=multiprocessing.cpu_count(), type=int, help="CPU Threads for transcoding")
    args = parser.parse_args()

    # Ask the user which transcode format they need
    choose_format = ['All', 'FLAC - 16bit', 'MP3 - 320', 'MP3 - V0', 'Only 320 and V0']
    selected_format = cli_ui.ask_choice("Select transcode formats", choices=choose_format)

    """
    Gather output folder naming information
    """
    ArtistName = []
    AlbumName = None
    Year = None
    bps = None
    basename = None
    flac_dir = os.path.abspath(args.path)
    output_dir = os.path.join(flac_dir, os.path.pardir)
    flac_files = locate(flac_dir, ext_matcher('.flac'))
    for flac_file in flac_files:
        flac_info = mutagen.flac.FLAC(flac_file)
        # print(flac_info.info.sample_rate)
        AlbumName = flac_info.get("album", 'Unknown Album')[0]
        ArtistName.append(flac_info.get("artist", 'Unknown Artist')[0])
        Year = flac_info.get("year", '0000')[0]
        bps = flac_info.info.bits_per_sample

    
    ArtistName = set(ArtistName)
    if ArtistName:
        if len(ArtistName) < 2:
            ArtistName = list(ArtistName)[0]
        elif len(ArtistName) < 3:
            ArtistName = ", ".join(ArtistName)
        else:
            ArtistName = 'Various Artists'
    
    basename = f"{ArtistName} - {AlbumName} - ({Year})"

    if selected_format:
        if selected_format == choose_format[0]:
            output_format = ['FLAC', '320', 'V0']
        elif selected_format == choose_format[1]:
            output_format = 'FLAC'
        elif selected_format == choose_format[2]:
            output_format = '320'
        elif selected_format == choose_format[3]:
            output_format = 'V0'
        elif selected_format == choose_format[4]:
            output_format = ['320', 'V0']
    
        if type(output_format) == list:
            for i in range(len(output_format)):
                transcode_release(flac_dir, output_dir, basename, output_format[i], max_threads=None)
        else:
            pass
            transcode_release(flac_dir, output_dir, basename, output_format, max_threads=None)

if __name__ == "__main__": main()
