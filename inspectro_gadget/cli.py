"""Command-line interface for the InSpectro-Gadget analysis pipeline."""

import argparse
import os
import sys

from inspectro_gadget.gadget import gadget
from inspectro_gadget.version import __version__


def _flatten_args(items):
    """Normalize comma- or space-separated arguments into a flat list of strings."""
    if items is None:
        return None
    flattened = []
    for item in items:
        if isinstance(item, str) and "," in item:
            parts = [p.strip() for p in item.split(",") if p.strip()]
            flattened.extend(parts)
        else:
            flattened.append(item)
    return flattened


def create_parser():
    """Build the command-line argument parser with git-style subcommands."""
    parser = argparse.ArgumentParser(
        prog="inspectro-gadget",
        description="InSpectro-Gadget: Neurotransmitter and neuromodulator receptor gene expression pipeline for MRS voxels.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="subcommands",
        description="Analysis mode (run '<subcommand> --help' for detailed options)",
    )

    # 1. Single region
    region_parser = subparsers.add_parser(
        "region",
        help="Run receptor expression pipeline for a single region of interest.",
        description="Analyse receptor and neuromodulator gene expression within a single region of interest.",
    )
    region_parser.add_argument(
        "-m",
        "--mask-inputs",
        required=True,
        nargs="+",
        help="Path to region mask NIfTI image (.nii or .nii.gz in MNI152 2mm space).",
    )
    region_parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        default=None,
        help="Label for the region (defaults to 'Region 1').",
    )
    region_parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="Directory to save the PDF report and CSV tables (defaults to timestamped folder in current directory).",
    )
    region_parser.add_argument(
        "-b",
        "--background",
        default=None,
        help="Optional background NIfTI image path (MNI152 2mm space).",
    )

    # 2. Two-region comparison
    compare_parser = subparsers.add_parser(
        "compare",
        help="Run comparative receptor expression pipeline between two regions of interest.",
        description="Compare receptor and neuromodulator gene expression between two regions of interest.",
    )
    compare_parser.add_argument(
        "-m",
        "--mask-inputs",
        required=True,
        nargs="+",
        help="Paths to two region mask NIfTI images (.nii or .nii.gz in MNI152 2mm space).",
    )
    compare_parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        default=None,
        help="Labels for the two regions (e.g. -l 'ROI A' 'ROI B'; defaults to 'Region 1' 'Region 2').",
    )
    compare_parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="Directory to save the PDF report and CSV tables.",
    )
    compare_parser.add_argument(
        "-b",
        "--background",
        default=None,
        help="Optional background NIfTI image path (MNI152 2mm space).",
    )

    # 3. Multiple participant comparison
    multiple_parser = subparsers.add_parser(
        "multiple",
        help="Run group-level receptor expression pipeline across multiple participants.",
        description="Analyse receptor expression similarity and variance across multiple participant masks.",
    )
    multiple_parser.add_argument(
        "-m",
        "--mask-inputs",
        required=True,
        nargs="+",
        help="Paths to two or more participant mask NIfTI images (.nii or .nii.gz in MNI152 2mm space).",
    )
    multiple_parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        default=None,
        help="Participant labels matching mask inputs (e.g. -l sub-01 sub-02; defaults to Subject 1, Subject 2, ...).",
    )
    multiple_parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="Directory to save the PDF report, overlap NIfTI image, and CSV tables.",
    )
    multiple_parser.add_argument(
        "-b",
        "--background",
        default=None,
        help="Optional background NIfTI image path (MNI152 2mm space).",
    )
    multiple_parser.add_argument(
        "--no-multi-violin",
        action="store_false",
        dest="multi_violin",
        default=True,
        help="Disable generation of multi-participant violin plots.",
    )

    return parser


def run_cli(argv=None):
    """Execute the command-line interface with the provided argument list."""
    parser = create_parser()

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 1

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    # Normalize inputs
    masks = _flatten_args(args.mask_inputs)
    labels = _flatten_args(args.labels)

    # Validate mask files existence
    for m in masks:
        if not os.path.isfile(m):
            print(f"Error: Mask file not found: {m}", file=sys.stderr)
            return 1

    # Validate background file existence if provided
    if args.background is not None and not os.path.isfile(args.background):
        print(f"Error: Background file not found: {args.background}", file=sys.stderr)
        return 1

    # Ensure output directory exists if specified
    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)

    try:
        if args.command == "region":
            if len(masks) != 1:
                print(
                    f"Error: Subcommand 'region' requires exactly 1 mask input, but received {len(masks)}.",
                    file=sys.stderr,
                )
                return 1
            if labels is not None and len(labels) != 1:
                print(
                    f"Error: Subcommand 'region' expects at most 1 label, but received {len(labels)}.",
                    file=sys.stderr,
                )
                return 1
            gadget(
                mask_fnames=[masks[0]],
                mask_labels=[labels[0]] if labels else None,
                out_root=args.outdir,
                bground_fname=args.background,
            )
            return 0

        elif args.command == "compare":
            if len(masks) != 2:
                print(
                    f"Error: Subcommand 'compare' requires exactly 2 mask inputs, but received {len(masks)}.",
                    file=sys.stderr,
                )
                return 1
            if labels is not None and len(labels) != 2:
                print(
                    f"Error: Subcommand 'compare' expects exactly 2 labels, but received {len(labels)}.",
                    file=sys.stderr,
                )
                return 1
            gadget(
                mask_fnames=[[masks[0]], [masks[1]]],
                mask_labels=list(labels) if labels else None,
                out_root=args.outdir,
                bground_fname=args.background,
            )
            return 0

        elif args.command == "multiple":
            if len(masks) < 2:
                print(
                    f"Error: Subcommand 'multiple' requires at least 2 mask inputs, but received {len(masks)}.",
                    file=sys.stderr,
                )
                return 1
            if labels is not None and len(labels) != len(masks):
                print(
                    f"Error: Subcommand 'multiple' received {len(labels)} labels for {len(masks)} mask inputs; count must match.",
                    file=sys.stderr,
                )
                return 1
            gadget(
                mask_fnames=list(masks),
                mask_labels=list(labels) if labels else None,
                out_root=args.outdir,
                bground_fname=args.background,
                multi_violin=args.multi_violin,
            )
            return 0

        else:
            print(f"Error: Unknown subcommand '{args.command}'.", file=sys.stderr)
            parser.print_help()
            return 1

    except Exception as err:
        print(f"Execution failed: {err}", file=sys.stderr)
        return 1


def main():
    """Main entry point for console scripts."""
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
